import openai
import os

from dotenv import load_dotenv

# Configured by environment variables
load_dotenv()


openai.api_key=os.getenv('OPENAI_API_KEY')
openai.base_url=os.getenv('OPENAI_BASE_URL')

client = openai.OpenAI()

sentences = [
    'Beyoncé lanzó su nuevo álbum sorpresa ayer.'
]

messages = [
    {"role": "user", "content": f"Translate the following spanish sentences to guarani: {sentences[0]}"},
]

chat_response = client.chat.completions.create(
    model="Qwen/Qwen3.5-9B",
    messages=messages, # type: ignore
    max_tokens=81920,
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
        "min_p": 0.0,
        "repetition_penalty":1.0
    }
)

print("Chat response:", chat_response.choices[0].message.content)