from transformers import AutoModelForCausalLM, AutoTokenizer

import re
import torch



def get_prompt(text):
    prompt = f"""
        Translate from spanish to guarani the following text. Provide a short
        translation (max 40 words) and output the translation enclosed in 
        <translation></translation>.
            
        Text: `{text}`
    """
    return prompt
    

model_name = "openai/gpt-oss-20b"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)


sentences = [
    'Beyoncé lanzó su nuevo álbum sorpresa ayer.',
    'El festival de jazz reunió a artistas de todo el mundo.',
    'La exposición de arte moderno abre mañana en Madrid.',
    'La gala de los Óscar será transmitida en directo',
    'Escuchar ópera en ese teatro siempre es una experiencia única',
    'El actor principal ganó un Goya este año',
    '‘Roma’ cautivó a la crítica internacional en 2018',
    'El mural presenta colores vibrantes y figuras surrealistas',
    '¿Viste el tráiler del próximo blockbuster de Marvel?',
    'La serie tiene ocho temporadas en total'
]

prompts = []
for sentence in sentences:
    # concatenate batch sentence to the task prompt
    task_prompt = get_prompt(sentence)
    messages = [
        {'role': 'system', 'content': 'Your are an expert translator from spanish to guarani.'},
        {'role': 'user', 'content': task_prompt},
    ]
    # apply the chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort='low'
    )
    prompts.append(prompt)

inputs = tokenizer(
    prompts, return_tensors='pt', padding=True, truncation=True, padding_side='left'
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=True
    )

for j in range(len(sentences)):
    #input_len = inputs['attention_mask'][j].sum()
    input_len = inputs['input_ids'].shape[1]
    generated_tokens = outputs[j, input_len:]
    raw_output = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=False
    ).strip()
    try:
        final_output = raw_output.split('|>final<|')[1]
        translation = re.findall(r'<translation>(.*?)</translation>', final_output, re.DOTALL)[-1].strip()
        print(translation)
    except:
        print(f'Problem in parsing the the raw output. Raw output: {raw_output}')
