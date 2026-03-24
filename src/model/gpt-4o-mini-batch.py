import json
import os
import time

from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()



model_name = "gpt-4o-mini"
deployment_name = "gpt-4o-mini-2-batch"

api_key = os.getenv('AZURE_API_KEY')


def get_batch_result(client, batch_response):
    output_file_id = batch_response.output_file_id
    if not output_file_id:
        output_file_id = batch_response.error_file_id
    if output_file_id:
        file_response = client.files.content(output_file_id)
        raw_responses = file_response.text.strip().split('\n')  
        for raw_response in raw_responses:  
            json_response = json.loads(raw_response)  
            formatted_json = json.dumps(json_response, indent=2)  
            print(formatted_json)
    return json_response


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


def upload_file(client):
    # Upload a file with a purpose of 'batch'
    # 'expires_after' is an optional parameter that can be set to a number between 
    # 1209600-2592000. This is equivalent to 14-30 days
    file = client.files.create(
        file=open('test-gpt4o-batch.jsonl', 'rb'), 
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


def main():
    client = create_client()
    #file_id = upload_file(client)
    #batch_id = submit_batch_job(client, file_id)
    batch_id = 'batch_c14e4427-001d-45d1-84de-b4d05be907c7'
    batch_response = check_batch_execution_status(client, batch_id)
    result = get_batch_result(client, batch_response)

if __name__ == '__main__':
    main()