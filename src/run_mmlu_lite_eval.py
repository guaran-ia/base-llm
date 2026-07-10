import click
import csv
import json
import math
import os
import re
import sys
import time
import torch

from collections import defaultdict
from datetime import datetime
from transformers import GenerationConfig
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


DEFAULT_CONFIG = os.path.join('exp', 'global_mmlu_lite', 'config.json')
DEFAULT_OUTPUT_DIR = os.path.join('outputs', 'global_mmlu_lite')
VALID_ANSWERS = {'A', 'B', 'C', 'D'}
PROMPT_LANGUAGES = {'gn', 'es', 'en'}
QUESTION_FIELDS = ('question', 'question_gn')
OPTION_FIELDS = {
    'a': ('option_a', 'option_a_gn'),
    'b': ('option_b', 'option_b_gn'),
    'c': ('option_c', 'option_c_gn'),
    'd': ('option_d', 'option_d_gn'),
}


def resolve_path(path: str, project_dir: str = PROJECT_DIR) -> str:
    """Return an absolute path rooted at the project when needed."""
    if os.path.isabs(path):
        return path
    return os.path.join(project_dir, path)


def project_relative_path(path: str, project_dir: str = PROJECT_DIR) -> str:
    """Return a project-relative path for logs when the path is under the project."""
    abs_project_dir = os.path.abspath(project_dir)
    abs_path = os.path.abspath(path)
    try:
        if os.path.commonpath([abs_project_dir, abs_path]) == abs_project_dir:
            return os.path.relpath(abs_path, abs_project_dir)
    except ValueError:
        pass
    return path


def read_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def read_experiment_config(config_path: str) -> Dict[str, Any]:
    config = read_json(config_path)
    required_keys = ['dataset', 'base_models', 'exclude', 'max_new_tokens', 'prompt_language']
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(
            f'Config file is missing required keys: {", ".join(missing_keys)}'
        )
    return config


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def is_invalid_option(value: Any) -> bool:
    """Return True when a multiple-choice option cannot be evaluated."""
    if not isinstance(value, str):
        if isinstance(value, float) and math.isnan(value):
            return True
        return True
    stripped = value.strip()
    return not stripped or stripped.lower() == 'nan'


def first_present_value(row: Dict[str, Any], field_names: Iterable[str]) -> Any:
    """Return the first present field value from a row."""
    for field_name in field_names:
        if field_name in row:
            return row[field_name]
    return None


def get_question(row: Dict[str, Any]) -> Any:
    """Return the question text from supported Global MMLU-Lite schemas."""
    return first_present_value(row, QUESTION_FIELDS)


def get_option(row: Dict[str, Any], letter: str) -> Any:
    """Return an option value from supported Global MMLU-Lite schemas."""
    return first_present_value(row, OPTION_FIELDS[letter.lower()])


def validate_mmlu_row(row: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate the fields needed for four-option MMLU-style evaluation."""
    errors = []
    if not row.get('sample_id'):
        errors.append('missing sample_id')
    question = get_question(row)
    if not isinstance(question, str) or not question.strip():
        errors.append('missing question')
    if row.get('answer') not in VALID_ANSWERS:
        errors.append('invalid answer')
    for letter in 'abcd':
        if is_invalid_option(get_option(row, letter)):
            errors.append(f'invalid option_{letter}')
    return len(errors) == 0, errors


def split_valid_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Separate valid rows from rows skipped by the invalid-option policy."""
    valid_rows = []
    skipped_rows = []
    for index, row in enumerate(rows, start=1):
        is_valid, errors = validate_mmlu_row(row)
        if is_valid:
            valid_rows.append(row)
        else:
            skipped_rows.append({
                'line': index,
                'sample_id': row.get('sample_id'),
                'subject': row.get('subject'),
                'subject_category': row.get('subject_category'),
                'answer': row.get('answer'),
                'errors': errors,
            })
    return valid_rows, skipped_rows


def flatten_model_variants(base_models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flatten grouped HF model config into variant records."""
    variants = []
    for group in base_models:
        for variant in group.get('variants', []):
            huggingface_id = variant.get('huggingface_id')
            if not huggingface_id:
                raise ValueError(
                    'Global MMLU-Lite base model entries must contain only '
                    f'Hugging Face variants. Invalid entry: {variant}'
                )
            variants.append({
                'group_name': group.get('name', ''),
                'variant_name': variant.get('name', ''),
                'huggingface_id': huggingface_id,
                'model_name': huggingface_id.split('/')[-1].lower(),
            })
    return variants


def parse_excludes(values: Iterable[str]) -> set[str]:
    excludes = set()
    for value in values:
        for item in value.split(','):
            item = item.strip().lower()
            if item:
                excludes.add(item)
    return excludes


def filter_models(models: List[Dict[str, str]], excludes: set[str]) -> List[Dict[str, str]]:
    if not excludes:
        return models
    filtered = []
    for model in models:
        names = {
            model['model_name'].lower(),
            model['huggingface_id'].lower(),
            model['variant_name'].lower(),
        }
        if names.isdisjoint(excludes):
            filtered.append(model)
    return filtered


def sanitize_filename(name: str) -> str:
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    return safe_name.strip('._-').lower() or 'model'


def build_prompt(row: Dict[str, Any], prompt_language: str = 'gn') -> str:
    """Build the zero-shot Guarani Global MMLU-Lite prompt."""
    prompt_language = prompt_language.lower()
    if prompt_language not in PROMPT_LANGUAGES:
        raise ValueError(
            f'Unsupported prompt language: {prompt_language}. '
            f'Expected one of: {", ".join(sorted(PROMPT_LANGUAGES))}'
        )

    if prompt_language == 'es':
        instruction = 'Elige la respuesta correcta. Responde con una sola letra: A, B, C o D.'
        question_label = 'Pregunta'
        answer_label = 'Respuesta'
    elif prompt_language == 'en':
        instruction = 'Choose the correct answer. Reply with exactly one letter: A, B, C, or D.'
        question_label = 'Question'
        answer_label = 'Answer'
    else:
        instruction = 'Eiporavo mbohovái hekopete. Emyengovia peteĩ tai añoite: A, B, C térã D.'
        question_label = 'Porandu'
        answer_label = 'Mbohovái'

    return (
        f'{instruction}\n\n'
        f'{question_label}: {get_question(row).strip()}\n\n'
        f'A. {get_option(row, "a").strip()}\n'
        f'B. {get_option(row, "b").strip()}\n'
        f'C. {get_option(row, "c").strip()}\n'
        f'D. {get_option(row, "d").strip()}\n\n'
        f'{answer_label}:'
    )


def build_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {
            'role': 'system',
            'content': 'You answer multiple-choice questions. Return exactly one letter: A, B, C, or D.',
        },
        {'role': 'user', 'content': prompt},
    ]


def parse_answer(text: Optional[str]) -> Optional[str]:
    """Extract the first valid answer letter from model output."""
    if not text:
        return None
    stripped = text.strip()
    if stripped.upper() in VALID_ANSWERS:
        return stripped.upper()

    patterns = [
        r'(?:answer|respuesta|mbohov[aá]i)\s*(?:correcta|hekopete)?\s*(?:es|ha\'?e|:)?\s*[\(\[]?([ABCD])[\)\].,:;]?',
        r'(?:option|opci[oó]n)\s*[\(\[]?([ABCD])[\)\].,:;]?',
        r'^\s*([ABCD])\s*(?:thought|thinking\s+process|thinking|reasoning)\b',
        r'\b([ABCD])\b',
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def apply_chat_template(tokenizer: Any, prompt: str) -> str:
    messages = build_messages(prompt)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        system = messages[0]['content']
        user = messages[1]['content']
        return f'System: {system}\nUser: {user}\nAssistant:'


def load_hf_model(model_id: str) -> Tuple[Any, Any]:
    """Load one Hugging Face causal LM and tokenizer lazily."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hf_token = os.getenv('HF_ACCESS_TOKEN') or None
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side='left',
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        device_map=None,
        token=hf_token,
    ).to(device) # type: ignore
    model.eval()
    return model, tokenizer


def generate_batch(model: Any, tokenizer: Any, prompts: List[str], max_new_tokens: int) -> List[str]:
    """Generate model outputs for one batch of prompts."""
    rendered_prompts = [apply_chat_template(tokenizer, prompt) for prompt in prompts]
    inputs = tokenizer(
        rendered_prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
    ).to(model.device)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    stop_tokens = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids('<|eot_id|>'),
        tokenizer.convert_tokens_to_ids('<|im_end|>'),
        tokenizer.convert_tokens_to_ids('<|assistant_end|>'),
        tokenizer.convert_tokens_to_ids('</s>'),
    ]
    stop_tokens = [token for token in stop_tokens if token is not None and token != tokenizer.unk_token_id]
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=stop_tokens if stop_tokens else tokenizer.eos_token_id,
    )

    original_model_max_length = getattr(model.generation_config, 'max_length', None)
    try:
        model.generation_config.max_length = None
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
            )
    finally:
        model.generation_config.max_length = original_model_max_length

    decoded_outputs = []
    input_len = inputs['input_ids'].shape[1]
    stop_strings = ['<extra_id_1>', '<|im_end|>', '<|assistant_end|>', '<|eot_id|>', '</s>']
    for output in outputs:
        generated_tokens = output[input_len:]
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        for stop_string in stop_strings:
            if stop_string in text:
                text = text.split(stop_string)[0].strip()
        decoded_outputs.append(text.replace('\n', ' ').strip())
    return decoded_outputs


def aggregate_results(predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute overall, category, and subject accuracies."""
    total = len(predictions)
    correct = sum(1 for row in predictions if row['is_correct'])
    null_predictions = sum(1 for row in predictions if row.get('prediction') is None)
    by_category = defaultdict(lambda: {'total': 0, 'correct': 0})
    by_subject = defaultdict(lambda: {'total': 0, 'correct': 0})

    for row in predictions:
        category = row.get('subject_category') or 'unknown'
        subject = row.get('subject') or 'unknown'
        by_category[category]['total'] += 1
        by_subject[subject]['total'] += 1
        if row['is_correct']:
            by_category[category]['correct'] += 1
            by_subject[subject]['correct'] += 1

    return {
        'total': total,
        'correct': correct,
        'null_predictions': null_predictions,
        'accuracy': correct / total if total else 0.0,
        'accuracy_by_subject_category': format_group_scores(by_category),
        'accuracy_by_subject': format_group_scores(by_subject),
    }


def format_group_scores(group_scores: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, Any]]:
    formatted = {}
    for group_name, scores in sorted(group_scores.items()):
        total = scores['total']
        correct = scores['correct']
        formatted[group_name] = {
            'total': total,
            'correct': correct,
            'accuracy': correct / total if total else 0.0,
        }
    return formatted


def build_prediction_rows(rows: List[Dict[str, Any]], raw_outputs: List[str]) -> List[Dict[str, Any]]:
    predictions = []
    for row, raw_output in zip(rows, raw_outputs):
        prediction = parse_answer(raw_output)
        gold = row['answer']
        predictions.append({
            'sample_id': row['sample_id'],
            'subject': row.get('subject'),
            'subject_category': row.get('subject_category'),
            'question': get_question(row),
            'option_a': get_option(row, 'a'),
            'option_b': get_option(row, 'b'),
            'option_c': get_option(row, 'c'),
            'option_d': get_option(row, 'd'),
            'gold_answer': gold,
            'prediction': prediction,
            'is_correct': prediction == gold,
            'raw_output': raw_output,
        })
    return predictions


def evaluate_model(
    model_config: Dict[str, str],
    rows: List[Dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    prompt_language: str,
) -> List[Dict[str, Any]]:
    """Run one HF model over all valid evaluation rows."""
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda iterable, **_: iterable

    model, tokenizer = load_hf_model(model_config['huggingface_id'])
    prompts = [build_prompt(row, prompt_language) for row in rows]
    raw_outputs = []
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=f'Evaluating {model_config["model_name"]}',
    ):
        batch_prompts = prompts[start:start + batch_size]
        raw_outputs.extend(generate_batch(model, tokenizer, batch_prompts, max_new_tokens))

    try:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return build_prediction_rows(rows, raw_outputs)


def write_overall_csv(path: str, summaries: List[Dict[str, Any]]) -> None:
    category_names = sorted({
        category
        for summary in summaries
        for category in summary.get('accuracy_by_subject_category', {})
    })
    fieldnames = [
        'model_name',
        'huggingface_id',
        'status',
        'total',
        'correct',
        'null_predictions',
        'accuracy',
        'error',
    ] + [f'accuracy_category_{sanitize_filename(category)}' for category in category_names]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            row = {field: '' for field in fieldnames}
            row.update({ # type: ignore
                'model_name': summary.get('model_name'),
                'huggingface_id': summary.get('huggingface_id'),
                'status': summary.get('status'),
                'total': summary.get('total', 0),
                'correct': summary.get('correct', 0),
                'null_predictions': summary.get('null_predictions', 0),
                'accuracy': summary.get('accuracy', 0.0),
                'error': summary.get('error', ''),
            })
            category_scores = summary.get('accuracy_by_subject_category', {})
            for category in category_names:
                key = f'accuracy_category_{sanitize_filename(category)}'
                if category in category_scores:
                    row[key] = category_scores[category]['accuracy']
            writer.writerow(row)


def run_evaluation(
    dataset_path: str,
    base_models_path: str,
    config_path: str,
    output_dir: str,
    batch_size: int,
    excludes: set[str],
    max_samples: Optional[int],
    max_new_tokens: int,
    prompt_language: str,
) -> str:
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    run_dir = os.path.join(output_dir, f'{prompt_language}_{timestamp}') 
    os.makedirs(run_dir, exist_ok=True)

    rows = read_jsonl(dataset_path)
    valid_rows, skipped_rows = split_valid_rows(rows)
    if max_samples is not None:
        valid_rows = valid_rows[:max_samples]

    base_models = read_json(base_models_path)
    all_models = flatten_model_variants(base_models)
    selected_models = filter_models(all_models, excludes)

    metadata = {
        'timestamp': timestamp,
        'protocol': 'zero-shot-global-mmlu-lite-guarani',
        'config_path': project_relative_path(config_path),
        'dataset_path': project_relative_path(dataset_path),
        'base_models_path': project_relative_path(base_models_path),
        'output_dir': project_relative_path(run_dir),
        'batch_size': batch_size,
        'max_samples': max_samples,
        'max_new_tokens': max_new_tokens,
        'prompt_language': prompt_language,
        'invalid_option_policy': 'skip',
        'input_rows': len(rows),
        'evaluated_rows': len(valid_rows),
        'skipped_rows': skipped_rows,
        'excluded_models': sorted(excludes),
        'selected_models': selected_models,
        'model_errors': [],
    }
    write_json(os.path.join(run_dir, 'run_metadata.json'), metadata)

    summaries = []
    for model_config in selected_models:
        model_name = model_config['model_name']
        summary_path = os.path.join(run_dir, f'summary_{sanitize_filename(model_name)}.json')
        predictions_path = os.path.join(run_dir, f'predictions_{sanitize_filename(model_name)}.jsonl')
        started_at = time.time()
        try:
            predictions = evaluate_model(
                model_config,
                valid_rows,
                batch_size,
                max_new_tokens,
                prompt_language,
            )
            write_jsonl(predictions_path, predictions)
            summary = aggregate_results(predictions)
            summary.update({
                'model_name': model_name,
                'huggingface_id': model_config['huggingface_id'],
                'status': 'ok',
                'predictions_path': project_relative_path(predictions_path),
                'skipped_rows_count': len(skipped_rows),
                'duration_seconds': time.time() - started_at,
            })
        except Exception as exc:
            summary = {
                'model_name': model_name,
                'huggingface_id': model_config['huggingface_id'],
                'status': 'error',
                'total': 0,
                'correct': 0,
                'null_predictions': 0,
                'accuracy': 0.0,
                'error': str(exc),
                'skipped_rows_count': len(skipped_rows),
                'duration_seconds': time.time() - started_at,
            }
            metadata['model_errors'].append(summary)
        write_json(summary_path, summary)
        summaries.append(summary)

        metadata['completed_models'] = [summary['model_name'] for summary in summaries]
        write_json(os.path.join(run_dir, 'run_metadata.json'), metadata)

    overall_path = os.path.join(run_dir, f'overall_evaluation_{timestamp}.csv')
    write_overall_csv(overall_path, summaries)
    metadata['overall_evaluation_path'] = project_relative_path(overall_path)
    write_json(os.path.join(run_dir, 'run_metadata.json'), metadata)
    return run_dir


def run_cli(
    config: str,
    output_dir: str,
    batch_size: int,
    max_samples: Optional[int],
) -> str:
    """Validate CLI args and run the evaluation."""
    if batch_size <= 0:
        raise ValueError('--batch-size must be greater than 0')
    if max_samples is not None and max_samples <= 0:
        raise ValueError('--max-samples must be greater than 0')

    config_path = resolve_path(config)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    exp_config = read_experiment_config(config_path)

    dataset = exp_config['dataset']
    base_models = exp_config['base_models']
    exclude = exp_config['exclude']
    max_new_tokens = exp_config['max_new_tokens']
    prompt_language = exp_config['prompt_language']

    if max_new_tokens <= 0:
        raise ValueError('Config max_new_tokens must be greater than 0')
    prompt_language = prompt_language.lower()
    if prompt_language not in PROMPT_LANGUAGES:
        raise ValueError(
            f'Config prompt_language must be one of: {", ".join(sorted(PROMPT_LANGUAGES))}'
        )
    if isinstance(exclude, str):
        exclude_values = (exclude,)
    elif isinstance(exclude, list):
        exclude_values = exclude
    else:
        raise ValueError('Config exclude must be a list of strings or a comma-separated string')

    dataset_path = resolve_path(dataset)
    base_models_path = resolve_path(base_models)
    output_dir_path = resolve_path(output_dir)
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f'Dataset file not found: {dataset_path}')
    if not os.path.isfile(base_models_path):
        raise FileNotFoundError(f'Base models file not found: {base_models_path}')

    return run_evaluation(
        dataset_path=dataset_path,
        base_models_path=base_models_path,
        config_path=config_path,
        output_dir=output_dir_path,
        batch_size=batch_size,
        excludes=parse_excludes(exclude_values),
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
        prompt_language=prompt_language,
    )

@click.command()
@click.option('--config', default=DEFAULT_CONFIG, show_default=True)
@click.option('--output-dir', default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option('--batch-size', default=16, show_default=True, type=int)
@click.option('--max-samples', default=None, type=int, help='Optional number of valid samples to evaluate.')
@click.option('--invalid-option-policy', default='skip', show_default=True, type=click.Choice(['skip']))
def main(config, output_dir, batch_size, max_samples, invalid_option_policy) -> None:
    """Evaluate HF models on the configured Guarani Global MMLU-Lite set."""
    try:
        run_dir = run_cli(
            config=config,
            output_dir=output_dir,
            batch_size=batch_size,
            max_samples=max_samples,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f'Global MMLU-Lite evaluation finished. Outputs: {run_dir}')


if __name__ == '__main__':
    main()
