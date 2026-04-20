import os

from dotenv import load_dotenv
from openai import AzureOpenAI


load_dotenv()

endpoint = 'https://guarania-maas.cognitiveservices.azure.com/'
model_name = 'gpt-5.4-mini'
deployment = 'gpt-5.4-mini'

api_key = os.getenv('AZURE_API_KEY')
api_version = '2024-12-01-preview'

client = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=api_key,
)

sentence = "A la galería se presentan obras de arte contemporáneo"

prompt = f"""
Tell me if the following text corresponds to a valid Guarani sentence. Answer only yes or no.
Text: {sentence}
"""

response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "You are a linguistic expert in guarani.",
        },
        {
            "role": "user",
            "content": prompt,
        }
    ],
    max_completion_tokens=16384,
    model=deployment
)

print(response.choices[0].message.content)