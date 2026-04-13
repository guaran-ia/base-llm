import ast
import click
import json
import os
import openai
import requests
import time
import torch

from corpus.src.pipeline.language_identifier.language_identifier import LanguageIdentifier
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from utils.utils import read_jsonl, write_jsonl


identifier = LanguageIdentifier(glotlid=True, fasttext=True, openlid=True)


load_dotenv()


def read_experiment_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)


def read_base_models(base_models_list_path):
    with open(base_models_list_path, 'r') as f:
        return json.load(f)


def get_system_prompt(to_lang='guaraní', to_lang_iso='gn', 
                      from_lang='español', from_lang_iso='es'):
    return f"""
        Eres un experto traductor de {from_lang} (iso 639-1: {from_lang_iso}) a 
        {to_lang} (iso 639-1: {to_lang_iso}) y viceversa.
    """.strip()

def get_system_prompt_en(to_lang='guarani', to_lang_iso='gn', 
                         from_lang='spanish', from_lang_iso='es'):
    return f"""
        You are an expert translator from {from_lang} to {to_lang}.
    """.strip()


def get_task_prompt(text, from_lang='español', to_lang='guaraní'):
    return f"""
        Traduce de {from_lang} a {to_lang}. Solo devuelve la traducción, sin 
        explicaciones."

        Texto: "{text}"
    """.strip()


def get_task_prompt_en(text, from_lang='spanish', to_lang='guarani'):
    return f"""
        Translate from {from_lang} to {to_lang} the following text. Output exactly 
        one line with the translation ONLY. No further comments, explanation, 
        description, or thoughts are needed.
        
        Text: `{text}`
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


def sanitize_prompt(prompt_text):
    return ' '.join([pt for pt in prompt_text.replace('\n', ' ').split(' ') if pt])


def load_model(model_id):
    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side='left'
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None
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
            max_new_tokens=40,
            pad_token_id=tokenizer.pad_token_id
        )
    translated_text = tokenizer.decode(
        output[0][inputs['input_ids'].shape[-1]:],
        skip_special_tokens=True
    )
    end_time = time.time()
    duration = end_time - start_time
    return translated_text, duration


def do_batch_translation(model, tokenizer, sys_prompt, sentences, from_lang, 
                         to_lang, batch_size):
    max_new_tokens = 50
    trans_results = []
    loop_desc = f'Translating in batches sentences to {to_lang}'
    for i in tqdm(range(0, len(sentences), batch_size), desc=loop_desc):
        start_time = time.time()
        batch = sentences[i:i + batch_size]
        prompts = []
        for sentence in batch:
            # concatenate batch sentence to the task prompt
            task_prompt = sanitize_prompt(
                get_task_prompt_en(sentence, from_lang, to_lang)
            )
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
            prompts.append(prompt)
        # apply tokenization and move the input tensors to the same device as the model
        inputs = tokenizer(
            prompts, return_tensors='pt', padding=True, truncation=True
        ).to(model.device)
        # generate translation
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        # add tokens that tell the model to stop generating
        stop_tokens = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids('<|eot_id|>'),
            tokenizer.convert_tokens_to_ids('<|im_end|>'),
            tokenizer.convert_tokens_to_ids('<|assistant_end|>'),
            tokenizer.convert_tokens_to_ids('</s>')
        ]
        stop_tokens = [t for t in stop_tokens if t is not None]
        stop_strings = [
            '<extra_id_1>',
            '<|im_end|>',
            '<|assistant_end|>',
            '<|eot_id|>',
            '</s>'
        ]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=stop_tokens,
                stop_strings=stop_strings,
                tokenizer=tokenizer
            )
        # process output
        for j in range(len(batch)):
            #input_len = inputs['attention_mask'][j].sum()
            input_len = inputs['input_ids'].shape[1]
            generated_tokens = outputs[j, input_len:]
            translation = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()
            # remove stop tokens
            for stop in stop_strings:
                if stop in translation:
                    translation = translation.split(stop)[0]
            translation = translation.replace('\n', ' ').strip()
            trans_results.append(translation)
        end_time = time.time()
        duration = end_time - start_time
        #print(f'Translation of {batch_size} sentences lasted: {duration} secs')
    return trans_results


def get_lang_translation(translation):
    result = identifier.identify_languages(translation, k=1)
    if result is not None and \
       'languages' in result and \
       result['languages'] is not None and \
       len(result['languages']) > 0:
        return result['languages'][0]
    else:
        return ''


def do_run_batch_rtt(rtt_data, model_variant, sys_prompt, from_lang, from_lang_iso,
                     to_lang, to_lang_iso, batch_size):
    model_variant_id = model_variant['huggingface_id']
    model, tokenizer = load_model(model_variant_id)
    sentences = [record['text'] for record in rtt_data]
    # batch translate to language (e.g., guarani)
    forward_trans = do_batch_translation(model, tokenizer, sys_prompt, sentences, 
                                         from_lang, to_lang, batch_size)
    assert len(forward_trans) == len(sentences), \
        f'The numer of forward translations ({len(forward_trans)}) is '\
        f'inconsistent with the number of sentences ({len(sentences)})'
    # batch translate back to language (e.g., spanish)
    backward_trans = do_batch_translation(model, tokenizer, sys_prompt, forward_trans, 
                                          to_lang, from_lang, batch_size)
    assert len(backward_trans) == len(sentences), \
        f'The numer of backward translations ({len(backward_trans)}) is '\
        f'inconsistent with the number of sentences ({len(sentences)})'
    rtt_model = {
        'model': {
            'name': model_variant['huggingface_id']
        },
        'params': {
            'from_lang': f'{from_lang} ({from_lang_iso})',
            'to_lang': f'{to_lang} ({to_lang_iso})'
        },
        'rtt_translation': []
    }
    for idx, record in enumerate(rtt_data):
        # check language of the forward and backward translation
        fwd_tran_lang = get_lang_translation(forward_trans[idx])
        bkw_trans_lang = get_lang_translation(backward_trans[idx])
        rtt_model['rtt_translation'].append(
            {
                'id': record['id'],
                f'source_text_{from_lang_iso}': record['text'],
                f'translated_{to_lang_iso}_text': forward_trans[idx],
                f'translated_{to_lang_iso}_language': fwd_tran_lang,
                f'translated_{from_lang_iso}_text': backward_trans[idx],
                f'translated_{from_lang_iso}_language': bkw_trans_lang
            }
        )
    return rtt_model
    

def do_run_rtt(rtt_data, model_variant, sys_prompt, from_lang, from_lang_iso,
               to_lang, to_lang_iso):
    trans_duration = []
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
        'rtt_translation': []
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
        rtt_model['rtt_translation'].append(
            {
                f'source_text_{from_lang_iso}': source_text,
                f'translated_{to_lang_iso}_text': trans_text_to_lang,
                f'translated_{from_lang_iso}_text': trans_text_from_lang
            }
        )
        if len(trans_duration) > 0 and len(trans_duration)%100 == 0:
            print(f'\nMean duration in secs per 100 translations={sum(trans_duration)/len(trans_duration)}')
    return rtt_model


def save_results(results, model_variant_name, output_dir):
    # save the results in output_dir
    with open(os.path.join(output_dir, f'{model_variant_name}_rtt_results.json'), 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def translate_qwen3_5(sys_prompt, task_prompt, client):
    messages = [
        {'role': 'system', 'content': sys_prompt},
        {'role': 'user', 'content': task_prompt},
    ]    
    chat_response = client.chat.completions.create(
        model='Qwen/Qwen3.5-9B',
        messages=messages, # type: ignore
        max_tokens=81920,
        temperature=0.7,
        top_p=0.8,
        presence_penalty=1.5,
        extra_body={
            'chat_template_kwargs': {'enable_thinking': False},
            'top_k': 20,
            'min_p': 0.0,
            'repetition_penalty':1.0
        }
    )
    if len(chat_response.choices) > 0:
        return chat_response.choices[0].message.content
    else:
        return ''


def do_run_rtt_qwen3_5(rtt_data, model_variant, sys_prompt, from_lang, 
                       from_lang_iso, to_lang, to_lang_iso):
    # set qwen requirements from environment variables
    openai.api_key=os.getenv('OPENAI_API_KEY')
    openai.base_url=os.getenv('OPENAI_BASE_URL')
    # instantiate openai client
    client = openai.OpenAI()
    # initialize object to save translations
    rtt_model = {
        'model': {
            'name': model_variant
        },
        'params': {
            'from_lang': f'{from_lang} ({from_lang_iso})',
            'to_lang': f'{to_lang} ({to_lang_iso})'
        },
        'rtt_translation': []
    }
    # iterate over sentences
    loop_desc = 'Translating sentences with Qwen 3.5...'
    for record in tqdm(rtt_data, desc=loop_desc):
        sentence = record['text']
        # translate to `to_lang` (e.g., guarani) using qwen
        task_prompt = sanitize_prompt(
            get_task_prompt_en(sentence, from_lang, to_lang)
        )
        forward_trans = translate_qwen3_5(sys_prompt, task_prompt, client)
        # translate to `from_lang` (e.g., spanish) using qwen
        backward_trans = ''
        if forward_trans:
            task_prompt = sanitize_prompt(
                get_task_prompt_en(forward_trans, to_lang, from_lang)
            )
            backward_trans = translate_qwen3_5(sys_prompt, task_prompt, client)
        # get language translations
        fwd_tran_lang = get_lang_translation(forward_trans)
        bkw_tran_lang = get_lang_translation(backward_trans)
        # save translations
        rtt_model['rtt_translation'].append(
            {
                'id': record['id'],
                f'source_text_{from_lang_iso}': sentence,
                f'translated_{to_lang_iso}_text': forward_trans,
                f'translated_{to_lang_iso}_language': fwd_tran_lang,
                f'translated_{from_lang_iso}_text': backward_trans,
                f'translated_{from_lang_iso}_language': bkw_tran_lang
            }
        )
    return rtt_model


def translate_grok(sys_prompt, task_prompt, model_id, end_point):
    headers = {
        'Authorization': f'Bearer {os.getenv("AZURE_API_KEY")}',
        'Content-Type': 'application/json'
    }
    data = {
        'model': model_id,
        'messages': [
            {'role': 'system', 'content': sys_prompt},
            {'role': 'user', 'content': task_prompt},
        ]
    }
    while True:
        response = requests.post(end_point, headers=headers, json=data)
        response_status_code = response.status_code
        if response_status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0 and \
               'message' in result['choices'][0] and \
               'content' in result['choices'][0]['message']:
                return result['choices'][0]['message']['content']
        time.sleep(1)
        print(f'Retrying translation after 1 second pause because response '\
              f'status code={response_status_code}')


def get_prompt_grok(from_lang, to_lang, sentences):
    return f"""
        Translate the following sentences from {from_lang} to {to_lang}. Respond 
        strictly in this format: [\"translation1\", \"translation2\", ...]. No 
        further comments, explanation, description, or thoughts are needed.

        Sentences
        {sentences}    
    """


def do_run_rtt_grok(rtt_data, model_id, sys_prompt, from_lang, from_lang_iso, 
                    to_lang, to_lang_iso, end_point, output_dir):
    
    # read grok translation file, if exist
    trans_file_path = os.path.join(output_dir, 'tmp_grok_translations.jsonl')
    tmp_trans_lookup = None
    if os.path.isfile(trans_file_path):
        tmp_trans = read_jsonl(trans_file_path)
        tmp_trans_lookup = {tt['id']: tt for tt in tmp_trans}
    # initialize object to save translations
    rtt_model = {
        'model': {
            'name': model_id
        },
        'params': {
            'from_lang': f'{from_lang} ({from_lang_iso})',
            'to_lang': f'{to_lang} ({to_lang_iso})'
        },
        'rtt_translation': []
    }
    # iterate over sentences
    loop_desc = f'Translating sentences with {model_id}...'
    for record in tqdm(rtt_data, desc=loop_desc):
        # do the translation only if the sentence hasn't translated yet
        if not tmp_trans_lookup or (tmp_trans_lookup and record['id'] not in tmp_trans_lookup):
            sentence = record['text']
            # perform forward sentence to `to_lang` (e.g., guarani)
            task_prompt = sanitize_prompt(get_task_prompt_en(sentence, from_lang, to_lang))
            forward_trans = translate_grok(sys_prompt, task_prompt, model_id, end_point)
            if forward_trans:
                # perform backward sentences to `from_lang` (e.g., spanish)
                task_prompt = sanitize_prompt(get_task_prompt_en(forward_trans, to_lang, from_lang))
                backward_trans = translate_grok(sys_prompt, task_prompt, model_id, end_point)
                fwd_tran_lang = get_lang_translation(forward_trans)
                bkw_trans_lang = get_lang_translation(backward_trans)
                trans_dict = {
                    'id': record['id'],
                    f'source_text_{from_lang_iso}': sentence,
                    f'translated_{to_lang_iso}_text': forward_trans,
                    f'translated_{to_lang_iso}_language': fwd_tran_lang,
                    f'translated_{from_lang_iso}_text': backward_trans,
                    f'translated_{from_lang_iso}_language': bkw_trans_lang
                }
                write_jsonl(trans_file_path, [trans_dict], mode='a')
                rtt_model['rtt_translation'].append(trans_dict)
            else:
                raise Exception(f'Forward translation is empty, stopping the process. '\
                                f'Sentence: {sentence}.')
    return rtt_model


def run_rtt(rtt_data, list_base_models, output_dir, to_lang, to_lang_iso, 
            from_lang, from_lang_iso, batch_size, models_to_exclude):
    sys_prompt = sanitize_prompt(
        get_system_prompt_en(to_lang, to_lang_iso, from_lang, from_lang_iso)
    )
    for base_model in tqdm(list_base_models, desc=f'Running RTT'):
        for model_variant in base_model['variants']:
            if 'huggingface_id' in model_variant:
                model_variant_name = model_variant['huggingface_id'].split('/')[-1].lower()
            else:
                model_variant_name = model_variant['model_id']
            if model_variant_name in models_to_exclude:
                continue
            print(f'\n\nRunning RTT with model: {model_variant_name}...')
            if model_variant_name == 'qwen3.5-9b':
                # Qwen 3.5 is treated differently since it is not invoked through
                # the Huggingface API but OpenAI's following the model documentation
                rtt_model = do_run_rtt_qwen3_5(
                    rtt_data, model_variant_name, sys_prompt, from_lang, from_lang_iso, 
                    to_lang, to_lang_iso
                )
            elif model_variant_name == 'grok-4-fast-non-reasoning':
                # Grok 4 has a special treatment since it is called through
                # the Azure API
                rtt_model = do_run_rtt_grok(
                    rtt_data, model_variant_name, sys_prompt, from_lang, from_lang_iso, 
                    to_lang, to_lang_iso, model_variant['end_point'], output_dir
                )
            else:
                if batch_size > 0:
                    rtt_model = do_run_batch_rtt(rtt_data, model_variant, sys_prompt, 
                                                from_lang, from_lang_iso, to_lang, 
                                                to_lang_iso, batch_size)
                else:
                    rtt_model = do_run_rtt(rtt_data, model_variant, sys_prompt,
                                        from_lang, from_lang_iso, to_lang, 
                                        to_lang_iso)
            # save results
            print(f'Saving RTT results...')
            save_results(rtt_model, model_variant_name, output_dir)
    print(f'RTT experiments have successfully finished!')


@click.command()
@click.option('--exp_dir', default='es_gn')
@click.option('--batch_size', default=64)
def main(exp_dir, batch_size):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_dir, 'data', 'rtt_experiments', exp_dir, 'config.json')
    # 0. load environment variables
    load_dotenv(os.path.join(project_dir, 'src', '.env'))
    # 1. login to HuggingFace Hub so we can access gated models, like gemma-3
    login(token=os.getenv('HF_ACCESS_TOKEN'))
    # 2. read experiment configuration
    print(f'Reading experiment configuration...')
    exp_config = read_experiment_config(config_path)
    # 3. create output directory (if it does not exist)
    output_dir = f'{exp_config["output_dir"]}_{datetime.now().strftime("%Y%m%d%H%M%S")}'
    output_dir_path = os.path.join(project_dir, output_dir)
    os.makedirs(output_dir_path, exist_ok=True)
    # 4. read RTT data
    rtt_data_path = os.path.join(project_dir, exp_config['rtt_data_path'])
    print(f'Reading dataset of sentences...')
    rtt_data = read_jsonl(rtt_data_path)
    print(f'In total, {len(rtt_data)} records were read')
    # 5. read list of base models
    print(f'Reading list of base models...')
    base_models_list_path = os.path.join(project_dir, exp_config['base_models_list_path'])
    list_base_models = read_base_models(base_models_list_path)
    # 6. run RTT for each base model and save the results in output_dir
    to_lang = exp_config['to_lang_en']
    to_lang_iso = exp_config['to_lang_iso']
    from_lang = exp_config['from_lang_en']
    from_lang_iso = exp_config['from_lang_iso']
    models_to_exclude = [m.lower() for m in exp_config['exclude']]
    run_rtt(rtt_data, list_base_models, output_dir_path, to_lang, to_lang_iso, 
            from_lang, from_lang_iso, batch_size, models_to_exclude)
    

if __name__ == '__main__':
    main()