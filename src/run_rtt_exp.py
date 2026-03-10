import torch
import json
import os
import time

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging
from tqdm import tqdm


# disable logging from transformers to avoid cluttering the output with warnings
# logging.set_verbosity_error()
# logging.disable_progress_bar()


load_dotenv()


def read_experiment_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def read_rtt_data(dataset_filepath):
    with open(dataset_filepath, 'r') as f:
        return [json.loads(line) for line in f]


def read_base_models(base_models_list_path):
    with open(base_models_list_path, 'r') as f:
        return json.load(f)


def get_system_prompt(to_lang='guaraní', to_lang_iso='gn', 
                      from_lang='español', from_lang_iso='es'):
    return f"""
        Eres un experto traductor de {from_lang} (iso 639-1: {from_lang_iso}) a 
        {to_lang} (iso 639-1: {to_lang_iso}) y viceversa.
    """


def get_task_prompt(text, from_lang='español', to_lang='guaraní'):
    return f"""
        Traduce el siguiente texto del {from_lang} al {to_lang}, manteniendo 
        el significado del texto original. El resultado debe ser solo la 
        traduccion.
        
        Texto a traducir: `{text}`
    """.strip()


def get_batch_task_prompt(from_lang='español', to_lang='guaraní'):
    return f"""
        Traduce los siguientes textos del {from_lang} al {to_lang}, manteniendo 
        el significado del texto original. Los textos a traducir se presentan a 
        continuación, uno por linea. El resultado debe ser solo las traducciones, 
        una por linea en el mismo orden. No repitas estas instrucciones.
        
        Textos a traducir:
    """.strip()


def get_text_prompt(text, from_lang, to_lang):
    return f"""
        El texto en {from_lang} a traducir a {to_lang} es: `{text}`
    """.strip()


def load_model(model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'Using device: {device}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16
    ).to(device) # type: ignore
    # put model in evaluation mode (inference)
    model.eval()
    return model, tokenizer


def do_translation(model, tokenizer, sys_prompt, task_prompt):
    start_time = time.time()
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': task_prompt},
    ]
    # apply the chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    # apply tokenization and move the input tensors to the same device as the model
    inputs = tokenizer(
        prompt, return_tensors='pt', padding=True, truncation=True
    ).to(model.device)
    # generate translation
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40
        )
    #translated_text = tokenizer.batch_decode(output, skip_special_tokens=True)
    translated_text = tokenizer.decode(
        output[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True
    )
    end_time = time.time()
    duration = end_time - start_time
    print(f'Translation lasted: {duration} secs')
    return translated_text, duration


def do_batch_translation(model, tokenizer, sys_prompt, task_prompt, sentences,
                         from_lang, to_lang, batch_size=256):
    
    trans_results = []
    for i in tqdm(range(0, len(sentences), batch_size), desc='Translating sentences in batches'):
        batch = sentences[i:i + batch_size]
        #content_prompt = task_prompt + '\n\n'
        prompts = []
        for sentence in batch:
            # concatenate batch sentence to the task prompt
            task_prompt = get_task_prompt(sentence, from_lang, to_lang)
            content_prompt = task_prompt + '\n\n' + f'`{sentence}`\n'    
            messages = [
                {'role': 'system', 'content': sys_prompt},
                {'role': 'user', 'content': content_prompt},
            ]
            # apply the chat template
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(prompt)
        # apply tokenization and move the input tensors to the same device as the model
        inputs = tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True, pad_to_multiple_of=8
        ).to(model.device)
        start_time = time.time()
        # generate translation
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=35,
                use_cache=True
            )
        print("Peak GPU memory (GB):", torch.cuda.max_memory_allocated() / 1e9)
        end_time = time.time()
        duration = end_time - start_time
        print(f'Translation of {batch_size} sentences lasted: {duration} secs')
        # process output
        for j in range(len(batch)):
            generated_tokens = outputs[j, inputs['input_ids'].shape[1]:]
            translation = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()
            trans_results.append(translation)
        
        # generated_tokens = outputs[0, inputs['input_ids'].shape[-1]:]
        # decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        # batch_translations = [trans.split() for trans in decoded.split('\n')]
        # if len(batch_translations) != batch_size:
        #     raise Exception(f'The number of translations ({len(batch_translations)}) '\
        #                     f'is inconsistent with the batch size {batch_size}. '\
        #                     f'Decoded: {decoded}')
        # trans_results.extend(batch_translations)

    
    return trans_results


def do_run_batch_rtt(rtt_data, model_variant, sys_prompt, from_lang, to_lang, 
                     batch_size):
    model_variant_id = model_variant['huggingface_id']
    model, tokenizer = load_model(model_variant_id)
    task_prompt = get_batch_task_prompt(from_lang, to_lang)
    sentences = [record['text'] for record in rtt_data]
    results = do_batch_translation(model, tokenizer, sys_prompt, task_prompt, sentences, 
                                   from_lang, to_lang, batch_size)
    
    

def do_run_rtt(rtt_data, model_variant, sys_prompt, from_lang, from_lang_iso,
               to_lang, to_lang_iso, output_dir):
    trans_duration = []
    results = []
    model_variant_id = model_variant['huggingface_id']
    model_variant_name = model_variant['huggingface_id'].split('/')[-1].lower()
    model, tokenizer = load_model(model_variant_id)
    rtt_model = {
        'model': {
            'name': model_variant['huggingface_id']
        },
        'params': {
            'from_lang': f'{from_lang} ({from_lang_iso})',
            'to_lang': f'{to_lang} ({to_lang_iso})'
        },
        'results': []
    }
    for record in tqdm(rtt_data, desc=f'Translating sentences with {model_variant_name}'):
        source_text = record['text']
        # translate to `to_lang` using the model
        task_prompt = get_task_prompt(source_text, from_lang, to_lang)
        trans_text_to_lang, duration = do_translation(
            model, tokenizer, sys_prompt, task_prompt
        )
        trans_duration.append(duration)
        trans_text_to_lang = trans_text_to_lang.replace('\n', ' ').strip()
        # translate back to `from_lang` using the model
        task_prompt = get_task_prompt(trans_text_to_lang, to_lang, from_lang)
        trans_text_from_lang, duration = do_translation(
            model, tokenizer, sys_prompt, task_prompt
        )
        trans_duration.append(duration)
        rtt_model['results'].append(
            {
                f'source_text_{from_lang_iso}': source_text,
                f'translated_{to_lang_iso}_text': trans_text_to_lang,
                f'translated_{from_lang_iso}_text': trans_text_from_lang
            }
        )
        if len(trans_duration) > 0 and len(trans_duration)%100 == 0:
            print(f'\nMean duration in secs per 100 translatos={sum(trans_duration)/len(trans_duration)}')
    # save the results in output_dir (not implemented here)
    with open(os.path.join(output_dir, f'{model_variant_name}_rtt_results.json'), 'w') as f:
        json.dump(rtt_model, f, ensure_ascii=False, indent=4)
    results.append(rtt_model)


def run_rtt(rtt_data, list_base_models, output_dir, to_lang='guaraní', to_lang_iso='gn', 
            from_lang='español', from_lang_iso='es', batch_mode=False):
    sys_prompt = get_system_prompt(to_lang, to_lang_iso, from_lang, from_lang_iso)
    results = []
    for base_model in tqdm(list_base_models, desc=f'Running RTT'):
        for model_variant in base_model['variants']:
            model_variant_name = model_variant['huggingface_id'].split('/')[-1].lower()
            if model_variant_name == 'gemma-3-4b-it':
                if not batch_mode:
                    rtt_model = do_run_rtt(rtt_data, model_variant, sys_prompt,
                                        from_lang, from_lang_iso, to_lang, 
                                        to_lang_iso, output_dir)
                    results.append(rtt_model)
                else:
                    batch_size = 256
                    do_run_batch_rtt(rtt_data, model_variant, sys_prompt, 
                                     from_lang, to_lang, batch_size)


def main(project_dir, exp_config_file_path):
    # 0. load environment variables
    load_dotenv(os.path.join(project_dir, 'src', '.env'))
    # 1. login to HuggingFace Hub so we can access gated models, like gemma-3
    login(token=os.getenv('HF_ACCESS_TOKEN'))
    # 2. read experiment configuration
    print(f'Reading experiment configuration...')
    exp_config = read_experiment_config(exp_config_file_path)
    # 3. create output directory (if it does not exist)
    output_dir = exp_config['output_dir']
    output_dir_path = os.path.join(project_dir, output_dir)
    os.makedirs(output_dir_path, exist_ok=True)
    # 4. read RTT data
    rtt_data_path = os.path.join(project_dir, exp_config['rtt_data_path'])
    print(f'Reading dataset of sentences...')
    rtt_data = read_rtt_data(rtt_data_path)
    print(f'In total, {len(rtt_data)} records were read')
    # 5. read list of base models
    print(f'Reading list of base models...')
    base_models_list_path = os.path.join(project_dir, exp_config['base_models_list_path'])
    list_base_models = read_base_models(base_models_list_path)
    # 6. run RTT for each base model and save the results in output_dir
    to_lang = exp_config['to_lang']
    to_lang_iso = exp_config['to_lang_iso']
    from_lang = exp_config['from_lang']
    from_lang_iso = exp_config['from_lang_iso']
    results = run_rtt(rtt_data, list_base_models, output_dir_path, to_lang, to_lang_iso, 
                      from_lang, from_lang_iso, batch_mode=True)


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_dir, 'data', 'rtt_experiments', 'es_gn', 'config.json')
    main(project_dir, config_path)