import click
import csv
import gc
import json
import os
import re
import sys
import time
import traceback

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)


DEFAULT_MODELS_CONFIG = os.path.join('exp', 'lm_eval', 'base_models.json')
DEFAULT_INCLUDE_PATH = os.path.join('exp', 'lm_eval')
DEFAULT_OUTPUT_DIR = os.path.join('outputs', 'lm_eval')
DEFAULT_TASK = 'gn_global_mmlu_lite'


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


def sanitize_filename(name: str) -> str:
    safe_name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    return safe_name.strip('._-').lower() or 'model'


def read_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, default=json_default)


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=json_default) + '\n')


def json_default(value: Any) -> Any:
    """Best-effort JSON fallback for numpy, tensors, and typed config objects."""
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def flatten_model_variants(base_models: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Flatten grouped HF model config into variant records."""
    variants = []
    for group in base_models:
        group_name = group.get('name', '')
        for variant in group.get('variants', []):
            huggingface_id = variant.get('huggingface_id')
            if not huggingface_id:
                raise ValueError(
                    'lm-eval base model entries must contain a huggingface_id. '
                    f'Invalid entry: {variant}'
                )
            variant_name = variant.get('name', '')
            run_name = sanitize_filename(f'{group_name}__{variant_name}')
            variants.append({
                'group_name': group_name,
                'variant_name': variant_name,
                'huggingface_id': huggingface_id,
                'model_name': huggingface_id.split('/')[-1].lower(),
                'run_name': run_name,
            })
    return variants


def parse_csv_values(values: Iterable[str]) -> set[str]:
    parsed = set()
    for value in values:
        for item in value.split(','):
            item = item.strip().lower()
            if item:
                parsed.add(item)
    return parsed


def model_aliases(model_config: Dict[str, str]) -> set[str]:
    return {
        model_config['group_name'].lower(),
        model_config['variant_name'].lower(),
        model_config['huggingface_id'].lower(),
        model_config['model_name'].lower(),
        model_config['run_name'].lower(),
    }


def select_models(
    models: List[Dict[str, str]],
    only: set[str],
    excludes: set[str],
) -> List[Dict[str, str]]:
    selected = []
    for model_config in models:
        aliases = model_aliases(model_config)
        if only and aliases.isdisjoint(only):
            continue
        if excludes and not aliases.isdisjoint(excludes):
            continue
        selected.append(model_config)
    return selected


def build_model_args(
    model_backend: str,
    huggingface_id: str,
    dtype: str,
    trust_remote_code: bool,
    enable_thinking: Optional[bool],
) -> Dict[str, Any]:
    if model_backend != 'hf':
        return {}

    model_args: Dict[str, Any] = {
        'pretrained': huggingface_id,
        'dtype': dtype,
        'trust_remote_code': trust_remote_code,
    }
    if enable_thinking is not None:
        model_args['enable_thinking'] = enable_thinking
    return model_args


def extract_metric(results: Optional[Dict[str, Any]], task: str, metric: str) -> Any:
    if not results:
        return None
    task_results = results.get('results', {}).get(task, {})
    if metric in task_results:
        return task_results[metric]

    metric_prefix = f'{metric},'
    for key, value in task_results.items():
        if key.startswith(metric_prefix):
            return value
    return None


def extract_stderr(results: Optional[Dict[str, Any]], task: str, metric: str) -> Any:
    if not results:
        return None
    task_results = results.get('results', {}).get(task, {})
    candidates = [
        f'{metric}_stderr',
        f'{metric}_stderr,none',
        f'{metric},none_stderr',
    ]
    for key in candidates:
        if key in task_results:
            return task_results[key]
    for key, value in task_results.items():
        if key.startswith(metric) and 'stderr' in key:
            return value
    return None


def write_summary_csv(path: str, summaries: List[Dict[str, Any]]) -> None:
    fieldnames = [
        'group_name',
        'variant_name',
        'huggingface_id',
        'run_name',
        'task',
        'status',
        'acc',
        'stderr',
        'started_at',
        'finished_at',
        'duration_seconds',
        'output_dir',
        'error',
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary.get(field, '') for field in fieldnames})


def clear_model_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    except Exception:
        pass


def run_one_model(
    model_config: Dict[str, str],
    run_dir: str,
    task: str,
    include_path: Optional[str],
    model_backend: str,
    device: str,
    batch_size: int,
    dtype: str,
    limit: Optional[float],
    log_samples: bool,
    apply_chat_template: bool,
    trust_remote_code: bool,
    enable_thinking: Optional[bool],
    bootstrap_iters: int,
) -> Dict[str, Any]:
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager

    model_output_dir = os.path.join(run_dir, model_config['run_name'])
    results_path = os.path.join(model_output_dir, 'results.json')
    summary_path = os.path.join(model_output_dir, 'summary.json')
    error_path = os.path.join(model_output_dir, 'error.json')
    os.makedirs(model_output_dir, exist_ok=True)

    started_at_dt = datetime.now()
    started_at = time.time()
    summary: Dict[str, Any] = {
        **model_config,
        'task': task,
        'status': 'running',
        'started_at': started_at_dt.isoformat(timespec='seconds'),
        'output_dir': project_relative_path(model_output_dir),
    }
    write_json(summary_path, summary)

    try:
        task_manager = TaskManager(include_path=include_path)
        model_args = build_model_args(
            model_backend=model_backend,
            huggingface_id=model_config['huggingface_id'],
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
        )
        results = evaluator.simple_evaluate(
            model=model_backend,
            model_args=model_args,
            tasks=[task],
            task_manager=task_manager,
            batch_size=batch_size,
            device=device,
            apply_chat_template=apply_chat_template,
            log_samples=log_samples,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
        )
        write_json(results_path, results)

        finished_at = time.time()
        summary.update({
            'status': 'ok',
            'acc': extract_metric(results, task, 'acc'),
            'stderr': extract_stderr(results, task, 'acc'),
            'finished_at': datetime.now().isoformat(timespec='seconds'),
            'duration_seconds': finished_at - started_at,
            'results_path': project_relative_path(results_path),
        })
        write_json(summary_path, summary)
        return summary
    except Exception as exc:
        finished_at = time.time()
        error = {
            'status': 'error',
            'error_type': type(exc).__name__,
            'error': str(exc),
            'traceback': traceback.format_exc(),
        }
        write_json(error_path, error)
        summary.update({
            'status': 'error',
            'error': str(exc),
            'error_type': type(exc).__name__,
            'finished_at': datetime.now().isoformat(timespec='seconds'),
            'duration_seconds': finished_at - started_at,
            'error_path': project_relative_path(error_path),
        })
        write_json(summary_path, summary)
        return summary
    finally:
        clear_model_cache()


def run_experiment(
    models_config: str,
    task: str,
    include_path: Optional[str],
    output_dir: str,
    model_backend: str,
    device: str,
    batch_size: int,
    dtype: str,
    limit: Optional[float],
    log_samples: bool,
    apply_chat_template: bool,
    trust_remote_code: bool,
    enable_thinking: Optional[bool],
    bootstrap_iters: int,
    skip_existing: bool,
    dry_run: bool,
    run_name: Optional[str],
    only: set[str],
    excludes: set[str],
) -> str:
    os.chdir(PROJECT_DIR)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    experiment_name = sanitize_filename(run_name) if run_name else f'{task}_{timestamp}'
    run_dir = os.path.join(output_dir, experiment_name)
    os.makedirs(run_dir, exist_ok=True)

    base_models = read_json(models_config)
    all_models = flatten_model_variants(base_models)
    selected_models = select_models(all_models, only=only, excludes=excludes)

    metadata = {
        'timestamp': timestamp,
        'task': task,
        'models_config': project_relative_path(models_config),
        'include_path': project_relative_path(include_path) if include_path else None,
        'output_dir': project_relative_path(run_dir),
        'model_backend': model_backend,
        'device': device,
        'batch_size': batch_size,
        'dtype': dtype,
        'limit': limit,
        'log_samples': log_samples,
        'apply_chat_template': apply_chat_template,
        'trust_remote_code': trust_remote_code,
        'enable_thinking': enable_thinking,
        'bootstrap_iters': bootstrap_iters,
        'skip_existing': skip_existing,
        'dry_run': dry_run,
        'run_name': experiment_name,
        'only': sorted(only),
        'exclude': sorted(excludes),
        'selected_models': selected_models,
    }
    metadata_path = os.path.join(run_dir, 'run_metadata.json')
    summary_jsonl_path = os.path.join(run_dir, 'summary.jsonl')
    summary_csv_path = os.path.join(run_dir, 'summary.csv')
    write_json(metadata_path, metadata)

    summaries: List[Dict[str, Any]] = []
    for model_config in selected_models:
        model_output_dir = os.path.join(run_dir, model_config['run_name'])
        summary_path = os.path.join(model_output_dir, 'summary.json')
        results_path = os.path.join(model_output_dir, 'results.json')

        if dry_run:
            os.makedirs(model_output_dir, exist_ok=True)
            summary = {
                **model_config,
                'task': task,
                'status': 'planned',
                'output_dir': project_relative_path(model_output_dir),
            }
            write_json(summary_path, summary)
        elif skip_existing and os.path.isfile(results_path):
            summary = read_json(summary_path) if os.path.isfile(summary_path) else {
                **model_config,
                'task': task,
                'status': 'skipped',
                'output_dir': project_relative_path(model_output_dir),
                'results_path': project_relative_path(results_path),
            }
        else:
            click.echo(
                f'[{len(summaries) + 1}/{len(selected_models)}] '
                f'Evaluating {model_config["huggingface_id"]}'
            )
            summary = run_one_model(
                model_config=model_config,
                run_dir=run_dir,
                task=task,
                include_path=include_path,
                model_backend=model_backend,
                device=device,
                batch_size=batch_size,
                dtype=dtype,
                limit=limit,
                log_samples=log_samples,
                apply_chat_template=apply_chat_template,
                trust_remote_code=trust_remote_code,
                enable_thinking=enable_thinking,
                bootstrap_iters=bootstrap_iters,
            )

        summaries.append(summary)
        metadata['completed_models'] = [
            summary['run_name'] for summary in summaries
        ]
        write_json(metadata_path, metadata)
        write_jsonl(summary_jsonl_path, summaries)
        write_summary_csv(summary_csv_path, summaries)

    return run_dir


def parse_limit(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    if value.strip() == '':
        return None
    parsed = float(value)
    if parsed <= 0:
        raise ValueError('--limit must be greater than 0')
    if parsed.is_integer():
        return int(parsed)
    if parsed > 1:
        raise ValueError('fractional --limit values must be <= 1')
    return parsed


def run_cli(
    models_config: str,
    task: str,
    include_path: Optional[str],
    output_dir: str,
    model_backend: str,
    device: str,
    batch_size: int,
    dtype: str,
    limit: Optional[str],
    log_samples: bool,
    apply_chat_template: bool,
    trust_remote_code: bool,
    enable_thinking: Optional[bool],
    bootstrap_iters: int,
    skip_existing: bool,
    dry_run: bool,
    run_name: Optional[str],
    only_values: Iterable[str],
    exclude_values: Iterable[str],
) -> str:
    if batch_size <= 0:
        raise ValueError('--batch-size must be greater than 0')
    if bootstrap_iters < 0:
        raise ValueError('--bootstrap-iters must be greater than or equal to 0')

    models_config_path = resolve_path(models_config)
    include_path_abs = resolve_path(include_path) if include_path else None
    output_dir_path = resolve_path(output_dir)
    if not os.path.isfile(models_config_path):
        raise FileNotFoundError(f'Model config file not found: {models_config_path}')
    if include_path_abs and not os.path.isdir(include_path_abs):
        raise FileNotFoundError(f'Include path not found: {include_path_abs}')

    return run_experiment(
        models_config=models_config_path,
        task=task,
        include_path=include_path_abs,
        output_dir=output_dir_path,
        model_backend=model_backend,
        device=device,
        batch_size=batch_size,
        dtype=dtype,
        limit=parse_limit(limit),
        log_samples=log_samples,
        apply_chat_template=apply_chat_template,
        trust_remote_code=trust_remote_code,
        enable_thinking=enable_thinking,
        bootstrap_iters=bootstrap_iters,
        skip_existing=skip_existing,
        dry_run=dry_run,
        run_name=run_name,
        only=parse_csv_values(only_values),
        excludes=parse_csv_values(exclude_values),
    )


@click.command()
@click.option('--models-config', default=DEFAULT_MODELS_CONFIG, show_default=True)
@click.option('--task', default=DEFAULT_TASK, show_default=True)
@click.option('--include-path', default=DEFAULT_INCLUDE_PATH, show_default=True)
@click.option('--output-dir', default=DEFAULT_OUTPUT_DIR, show_default=True)
@click.option('--model-backend', default='hf', show_default=True)
@click.option('--device', default='mps', show_default=True)
@click.option('--batch-size', default=1, show_default=True, type=int)
@click.option('--dtype', default='float16', show_default=True)
@click.option('--limit', default=None, help='Optional integer count or fraction for testing.')
@click.option('--log-samples/--no-log-samples', default=True, show_default=True)
@click.option('--apply-chat-template/--no-apply-chat-template', default=True, show_default=True)
@click.option('--trust-remote-code/--no-trust-remote-code', default=True, show_default=True)
@click.option('--enable-thinking/--disable-thinking', default=False, show_default=True)
@click.option('--bootstrap-iters', default=100000, show_default=True, type=int)
@click.option('--skip-existing', is_flag=True)
@click.option('--dry-run', is_flag=True, help='Write selected model plan without running lm-eval.')
@click.option('--run-name', default=None, help='Optional fixed output subdirectory name for resumable runs.')
@click.option('--only', 'only_values', multiple=True, help='Run only matching group, variant, run name, model name, or HF id.')
@click.option('--exclude', 'exclude_values', multiple=True, help='Skip matching group, variant, run name, model name, or HF id.')
def main(
    models_config,
    task,
    include_path,
    output_dir,
    model_backend,
    device,
    batch_size,
    dtype,
    limit,
    log_samples,
    apply_chat_template,
    trust_remote_code,
    enable_thinking,
    bootstrap_iters,
    skip_existing,
    dry_run,
    run_name,
    only_values,
    exclude_values,
) -> None:
    """Run lm-eval over every model variant in the configured model list."""
    try:
        run_dir = run_cli(
            models_config=models_config,
            task=task,
            include_path=include_path,
            output_dir=output_dir,
            model_backend=model_backend,
            device=device,
            batch_size=batch_size,
            dtype=dtype,
            limit=limit,
            log_samples=log_samples,
            apply_chat_template=apply_chat_template,
            trust_remote_code=trust_remote_code,
            enable_thinking=enable_thinking,
            bootstrap_iters=bootstrap_iters,
            skip_existing=skip_existing,
            dry_run=dry_run,
            run_name=run_name,
            only_values=only_values,
            exclude_values=exclude_values,
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc)) from exc
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f'lm-eval experiment finished. Outputs: {run_dir}')


if __name__ == '__main__':
    main()
