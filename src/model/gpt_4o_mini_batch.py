import click
import json
import os
import time

from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from src.utils.utils import read_jsonl, write_jsonl, write_json, read_json

load_dotenv()



model_name = 'gpt-4o-mini'
deployment_name = 'gpt-4o-mini-2-batch'

api_key = os.getenv('AZURE_API_KEY')


def get_batch_result(client, batch_response):
    output_file_id = batch_response.output_file_id
    if not output_file_id:
        output_file_id = batch_response.error_file_id
    if output_file_id:
        file_response = client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')  
        json_responses =[]
        for raw_response in raw_responses:  
            json_responses.append(json.loads(raw_response))
    return json_responses


def check_batch_execution_status(client, batch_id):
    status = 'validating'
    while status not in ('completed', 'failed', 'canceled'):
        time.sleep(60)
        batch_response = client.batches.retrieve(batch_id)
        status = batch_response.status
        print(f"{datetime.now()} Batch Id: {batch_id},  Status: {status}")

    if batch_response.status == "failed":
        if batch_response.errors is not None and batch_response.errors.data is not None:
            for error in batch_response.errors.data:  
                print(f"Error code {error.code} Message {error.message}")
    
    print(batch_response.model_dump_json(indent=2))
    return batch_response


def submit_batch_job(client, file_id):
    # Submit a batch job with the file
    batch_response = client.batches.create(
        input_file_id=file_id,
        endpoint='/chat/completions', #type: ignore # While passing this parameter is required, the system will read your input file to determine if the chat completions or responses API is needed.  
        completion_window='24h'
    )
    print(batch_response.model_dump_json(indent=2))
    # Save batch ID for later use
    return batch_response.id


def upload_file(client, file_path):
    # Upload a file with a purpose of 'batch'
    # 'expires_after' is an optional parameter that can be set to a number between 
    # 1209600-2592000. This is equivalent to 14-30 days
    file = client.files.create(
        file=open(file_path, 'rb'), 
        purpose='batch',
        extra_body={'expires_after':{'seconds': 1209600, 'anchor': 'created_at'}} 
    )
    print(file.model_dump_json(indent=2))
    print(f'File expiration: {datetime.fromtimestamp(file.expires_at) if file.expires_at is not None else "Not set"}')
    return file.id


def create_client():
    endpoint = "https://guarania-maas.cognitiveservices.azure.com/openai/v1/"
    client = OpenAI(
        base_url=f"{endpoint}",
        api_key=api_key
    )
    return client


def prepare_file_for_batch_gpt_4o_mini(rtt_data, from_lang_iso, to_lang_iso, from_lang, to_lang):
    # prepare the input file for batch translation with GPT-4o mini
    input_file_path = f'gpt_4o_mini_batch_input_{from_lang_iso}_{to_lang_iso}.jsonl'
    with open(input_file_path, 'w') as f:
        for record in rtt_data:
            sentence = record['text']
            json_line = json.dumps(
                {
                    'custom_id': f'task-{record['id']}', 
                    'method': 'POST',
                    'url': '/v1/chat/completions',
                    'body': {
                        'model': 'gpt-4o-mini-2-batch',
                        'messages': [
                            {
                                'role': 'system',
                                'content': f'You are an expert translator from {from_lang} to {to_lang}.'
                            },
                            {
                                'role': 'user',
                                'content': f'Translate the following sentence: {sentence}'
                            }
                        ]
                    }
                }
            )
            f.write(json_line + '\n')
    return input_file_path


def prepare_experiment_es_gn(project_dir, exp_dir):
    config_path = os.path.join(project_dir, 'data', 'rtt_experiments', exp_dir, 'config.json')
    exp_config = read_json(config_path)
    rtt_data_path = os.path.join(project_dir, exp_config['rtt_data_path']) # type: ignore
    rtt_data = read_jsonl(rtt_data_path)
    from_lang_iso = exp_config['from_lang_iso'] # type: ignore
    to_lang_iso = exp_config['to_lang_iso'] # type: ignore
    from_lang = exp_config['from_lang_en']
    to_lang = exp_config['to_lang_en']
    return prepare_file_for_batch_gpt_4o_mini(rtt_data, from_lang_iso, to_lang_iso, from_lang, to_lang)


def prepare_experiment_gn_es(project_dir, exp_dir, gn_translations_file_path):
    config_path = os.path.join(project_dir, 'data', 'rtt_experiments', exp_dir, 'config.json')
    exp_config = read_json(config_path)
    from_lang_iso = exp_config['to_lang_iso'] # type: ignore
    to_lang_iso = exp_config['from_lang_iso'] # type: ignore
    from_lang = exp_config['to_lang_en']
    to_lang = exp_config['from_lang_en']
    gn_translations_raw = read_jsonl(gn_translations_file_path)
    gn_translations = [{'id': t['id'], 'text': t['translation']} for t in gn_translations_raw if t['translation'] != '<translation_missing>']
    return prepare_file_for_batch_gpt_4o_mini(gn_translations, from_lang_iso, to_lang_iso, from_lang, to_lang)


@click.command()
@click.option('--target_lang', default='gn')
@click.option('--source_lang', default='es')
@click.option('--translations_file_path', default='')
def main(target_lang, source_lang, translations_file_path):
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    exp_dir = 'es_gn'
    if target_lang == 'gn' and source_lang == 'es':
        file_to_process = prepare_experiment_es_gn(project_dir, exp_dir)
    else:
        file_to_process = prepare_experiment_gn_es(project_dir, exp_dir, translations_file_path)
    client = create_client()
    file_id = upload_file(client, file_to_process)
    batch_id = submit_batch_job(client, file_id)
    batch_response = check_batch_execution_status(client, batch_id)
    result = get_batch_result(client, batch_response)
    output_dir = os.path.join(project_dir, 'outputs', 'rtt_experiment', f'gpt_4o_mini_batch_{datetime.now().strftime("%Y%m%d%H%M%S")}')
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f'gpt_4o_mini_batch_results_{exp_dir}.json')
    write_jsonl(output_file_path, result)


if __name__ == '__main__':
    main()