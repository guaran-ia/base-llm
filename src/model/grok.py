import ast
import os
import requests

from dotenv import load_dotenv

load_dotenv()


url = "https://guarania-maas.services.ai.azure.com/models/chat/completions?api-version=2024-05-01-preview"

api_key=os.getenv("AZURE_API_KEY")

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

prompt = """
    Translate the following spanish sentences to guarani. Respond strictly in 
    this format: [\"translation1\", \"translation2\", ...]
    
    Sentences
    1. El festival de jazz reunió a artistas de todo el mundo
    2. Beyoncé lanzó su nuevo álbum sorpresa ayer
    3. La exposición de arte moderno abre mañana en Madrid
    4. La gala de los Óscar será transmitida en directo
    5. Escuchar ópera en ese teatro siempre es una experiencia única
"""


data = {
    "model": "grok-4-fast-non-reasoning",
    "messages": [
        {"role": "user", "content": prompt}
    ]
}

response = requests.post(url, headers=headers, json=data)
result = response.json()
l_result = ast.literal_eval(result['choices'][0]['message']['content'])
for l in l_result:
    print(l)
