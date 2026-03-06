import torch
import json
import os

from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm

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


def get_task_prompt(from_lang='español', to_lang='guaraní'):
    return f"""
        Tu tarea es traducir el siguiente texto del {from_lang} al {to_lang}, 
        manteniendo el significado y el estilo del texto original. Asegúrate que 
        la traducción sea precisa y refleje el tono del texto original. Como 
        resultado, proporciona solo la traducción al {to_lang} sin ninguna 
        explicación, comentario, o texto adicional. Tampoco repitas estas 
        instrucciones en la salida.
    """.strip()


def get_text_prompt(text, from_lang, to_lang):
    return f"""
        El texto en {from_lang} a traducir a {to_lang} es: `{text}`
    """.strip()


def do_translation(model_id, sys_prompt, task_prompt, text_prompt):
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    
    pipe = pipeline(
        'text-generation',
        model=model_id,
        device=device,
        torch_dtype=torch.bfloat16,
    )
    
    #messages = [
    #    {'role': 'system', 'content': 'Eres un experto traductor de idiomas. Tu tarea es traducir el texto que se te proporciona de español a guarani, manteniendo el significado y el estilo del texto original. Proporciona solo la traducción sin ninguna explicación, comentario, o texto adicional.'},
    #    {'role': 'user', 'content': 'El festival de cannes se celebrara en mayo proximo'},
    #]
    
    #prompt = tokenizer.apply_chat_template(
    #    messages,
    #    tokenize=False,
    #    add_generation_prompt=True
    #)
    
    prompt = """
        Traduce el siguiente texto del español al guaraní.
        Responde únicamente con la traducción.

        Texto a traducir: `El festival de Cannes se celebrará en mayo próximo`
    """
    
    output = pipe(prompt, max_new_tokens=200)
    
    print(output[0]['generated_text'][len(prompt):]) 
    

def run_rtt(rtt_data, list_base_models, output_dir, to_lang='guaraní', to_lang_iso='gn', 
            from_lang='español', from_lang_iso='es'):
    sys_prompt = get_system_prompt(to_lang, to_lang_iso, from_lang, from_lang_iso)
    results = []
    for base_model in tqdm(list_base_models, desc=f'Running RTT for models'):
        for model_variant in base_model['variants']:
            model_variant_name = model_variant['huggingface_id'].split('/')[-1].lower()
            if model_variant_name == 'gemma-3-4b-it':
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
                    # translate to guaraní using the base model (not implemented here)
                    task_prompt = get_task_prompt(from_lang, to_lang)
                    text_prompt = get_text_prompt(source_text, from_lang, to_lang)
                    trans_text_to_lang = do_translation(
                        model_variant['huggingface_id'], sys_prompt, task_prompt, text_prompt
                    )
                    # translate back to spanish using the base model (not implemented here)
                    task_prompt = get_task_prompt(to_lang, from_lang)
                    text_prompt = get_text_prompt(trans_text_to_lang, to_lang, from_lang)
                    trans_text_from_lang = do_translation(
                        model_variant['huggingface_id'], sys_prompt, task_prompt, text_prompt
                    )
                    rtt_model['results'].append(
                        {
                            f'source_text_{from_lang_iso}': source_text,
                            f'translated_{to_lang_iso}_text': trans_text_to_lang,
                            f'translated_{from_lang_iso}_text': trans_text_from_lang
                        }
                    )
                # save the results in output_dir (not implemented here)
                with open(os.path.join(output_dir, f'{model_variant_name}_rtt_results.json'), 'w') as f:
                    json.dump(rtt_model, f, ensure_ascii=False, indent=4)
                results.append(rtt_model)
    return results


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
                      from_lang, from_lang_iso)


if __name__ == '__main__':
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_dir, 'data', 'rtt_experiments', 'es_gn', 'config.json')
    main(project_dir, config_path)