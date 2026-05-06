import ast
import click
import json
import os

from openai import AzureOpenAI, BadRequestError
from typing import Any, Dict, Iterator, List, Optional
from tqdm import tqdm
from src.utils.utils import get_rtt_config, read_json, write_json



def append_unsafe_content_log(text: str, reason: str, language_iso: str,
                              filename: str, project_dir: Optional[str] = None) -> None:
    """Append an unsafe-content event to a language-specific JSONL log."""
    log_dir = os.path.join(project_dir, 'outputs', 'log') # type: ignore
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f'unsafe_{language_iso}_content.jsonl')
    entry = {
        'unsafe_text': text,
        'reason': reason,
        'model': filename.split('_')[0] if '_' in filename else filename,
    }
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def get_content_filter_reason(error_results: Any) -> str:
    filtered_reasons = []
    for category, details in error_results.items():
        if isinstance(details, dict) and details.get('filtered'):
            severity = details.get('severity', 'unknown')
            filtered_reasons.append(f'{category}:{severity}')

    if filtered_reasons:
        return ', '.join(filtered_reasons)

    return 'Filtered by Azure content policy'


def get_azure_openai_client(project_dir: Optional[str] = None) -> 'AzureOpenAI':
    """Instantiate the Azure OpenAI client using project credentials."""
    if AzureOpenAI is None:
        raise ImportError('openai package is required for AzureOpenAI client creation')

    api_key = os.getenv('AZURE_OPENAI_API_KEY') or os.getenv('AZURE_API_KEY')
    if not api_key:
        raise RuntimeError('AZURE_OPENAI_API_KEY or AZURE_API_KEY is not set in environment or .env')

    azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    if not azure_endpoint:
        raise RuntimeError('AZURE_OPENAI_ENDPOINT is not set in environment or .env')

    api_version = os.getenv('AZURE_OPENAI_API_VERSION')
    if not api_version:
        raise RuntimeError('AZURE_OPENAI_API_VERSION is not set in environment or .env')

    return AzureOpenAI(
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        api_key=api_key,
    )


def build_language_validation_prompt(text: str, language_name: str) -> str:
    """Build a stable classification prompt for language validation."""
    return (
        f"Tell me if the following text corresponds to a valid {language_name} sentence.\n"
        "Answer only yes or no.\n"
        f"Text: {text}"
    )


def parse_yes_no_response(message: str) -> str:
    """Return a normalized yes/no label from a model response."""
    normalized = message.strip().lower()
    if normalized.startswith('yes'):
        return 'yes'
    if normalized.startswith('no'):
        return 'no'
    return '<error>'


def is_valid_language_sentence(text: str, language_name: str, language_iso: str,
                               filename: str, project_dir: Optional[str] = None) -> str:
    """Use Azure GPT-5.4 to verify whether text is a valid sentence in a given language."""
    if not text:
        return '<error>'
    client = get_azure_openai_client(project_dir)
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
    if not deployment:
        raise RuntimeError('AZURE_OPENAI_DEPLOYMENT is not set in environment or .env')
    prompt = build_language_validation_prompt(text, language_name)
    try:
        response = client.chat.completions.create(
            messages=[
                {'role': 'system', 'content': f'You are a linguistic expert in {language_name}.'},
                {'role': 'user', 'content': prompt},
            ],
            max_completion_tokens=64,
            model=deployment,
        )
        message = response.choices[0].message.content.strip().lower() #type: ignore
        parsed_message = parse_yes_no_response(message)
        if parsed_message != '<error>':
            return parsed_message
        print(
            f'Unexpected language-validation response for text: "{text}". '
            f'Response: "{message}"'
        )
        return 'no'
    except BadRequestError as e:
        error_text = str(e)
        payload_text = error_text.split(' - ')[-1]
        payload = ast.literal_eval(str(payload_text))
        error_dict = payload['error']
        if 'innererror' in error_dict and 'content_filter_result' in error_dict['innererror']:
            reason = get_content_filter_reason(error_dict['innererror']['content_filter_result'])
        else:
            reason = 'Filtered by Azure content policy'
        append_unsafe_content_log(text, reason, language_iso, filename, project_dir)
        print(f'Unsafe content detected for text: "{text}". Reason: {reason}. Logged to unsafe_{language_iso}_content.jsonl')
        return 'no'


def validate_translation_language(result_dir_name: str, project_dir: Optional[str] = None, 
                                  skip_existing: bool = True) -> None:
    """Annotate RTT result JSON files with valid_translated_<lang> flags."""
    config = get_rtt_config(result_dir_name, project_dir)
    from_lang_iso = config.get('from_lang_iso')
    to_lang_iso = config.get('to_lang_iso')
    from_lang_name = config.get(f'from_lang_en')
    to_lang_name = config.get(f'to_lang_en')
    
    results_dir = os.path.join(project_dir, 'outputs', 'rtt_experiment', result_dir_name) # type: ignore
    model_name = os.getenv('AZURE_OPENAI_DEPLOYMENT')

    if not os.path.isdir(results_dir):
        raise FileNotFoundError(f'Results directory not found: {results_dir}')

    for filename in os.listdir(results_dir):
        if not filename.endswith('.json'):
            continue
        file_path = os.path.join(results_dir, filename)
        model_results = read_json(file_path)
        updated = False
        print(f'\nProcessing {filename} using {model_name}...')
        model_name = filename.split('_')[0] if '_' in filename else filename
        desc_msg = f'Validating translations produced by {model_name}'
        for translation_dict in tqdm(model_results.get('rtt_translation', []), 
                                     desc=desc_msg):
            target_valid_key = f'valid_translated_{to_lang_iso}'
            source_valid_key = f'valid_translated_{from_lang_iso}'
            if skip_existing and target_valid_key in translation_dict and source_valid_key in translation_dict:
                continue

            translated_target = translation_dict.get(f'translated_{to_lang_iso}_text', '')
            translated_source = translation_dict.get(f'translated_{from_lang_iso}_text', '')

            translation_dict[target_valid_key] = \
                is_valid_language_sentence(
                    translated_target, to_lang_name, to_lang_iso, filename, project_dir  # type: ignore
                )
            translation_dict[source_valid_key] = \
                is_valid_language_sentence(
                    translated_source, from_lang_name, from_lang_iso, filename, project_dir # type: ignore
                )
            updated = True

        if updated:
            write_json(file_path, model_results)
            

@click.command()
@click.option('--res_dir', default='', help='Name of the directory containing the results (only the name, not the full path)')
def main(res_dir):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if res_dir:
        validate_translation_language(res_dir, project_dir, skip_existing=True)
    else:
        print('No result directory specified. Use --res_dir to specify the directory name containing the results to validate.')
    

if __name__ == "__main__":
    main()