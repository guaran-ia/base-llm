import click
import json
import os
import pandas as pd

from collections import defaultdict
from sacrebleu.metrics.bleu import BLEU
from sacrebleu.metrics.chrf import CHRF
from src.utils.utils import clean_text
from src.utils.utils import get_random_text
from src.utils.utils import iso_to_detection_code
from src.utils.utils import tokenize
from src.utils.utils import read_jsonl
from tqdm import tqdm



def compute_rtt_score(scores_dict):
    rtt_score = {}
    domain_score = []
    for domain, scores in scores_dict.items():
        if not scores:
            continue
        rtt_domain = sum(scores) / len(scores) if scores else 0.0
        rtt_score[domain.lower()] = rtt_domain
        domain_score.append(rtt_domain)
    rtt_score_general = sum(domain_score) / len(scores_dict) if domain_score else 0.0
    return rtt_score, rtt_score_general


def compute_actual_translations(translations, field_name, expected_code):
    actual = 0
    for translation_dict in translations:
        if translation_dict.get(field_name, '').lower() == expected_code:
            actual += 1
    return actual


def compute_num_valid_translations(translations, lang_code):
    valid = 0
    for translation in translations:
        if translation.get(f'valid_translated_{lang_code}', '') == 'yes':
            valid += 1
    return valid


def compute_num_language_disagreement(translations, lang_code):
    agreements = 0
    iso_code = iso_to_detection_code(lang_code)
    for translation in translations:
        if translation.get(f'translated_{lang_code}_language', '') == iso_code and \
           translation.get(f'valid_translated_{lang_code}', '') == 'yes':
            agreements += 1
    return len(translations) - agreements


def normalize_corpus_references(references):
    if not isinstance(references, list):
        raise TypeError('corpus evaluation references must be a list of strings')
    if all(isinstance(ref, str) for ref in references):
        return [references]
    if len(references) == 1 and isinstance(references[0], list) and all(isinstance(ref, str) for ref in references[0]):
        return references
    raise TypeError('corpus evaluation references must be either a list of strings or a list containing a single list of strings')


def evaluate_results(predictions, references, metric, mode='sentence'):
    if mode == 'sentence':
        eval_results = metric['obj'].sentence_score(predictions, references)
    else:
        eval_results = metric['obj'].corpus_score(predictions, normalize_corpus_references(references))
    return eval_results


def validate_translation(source, translation_tl, translation_fl, lang_forward_trans):
    if source == translation_tl:
        # if source and translation are equal, it means that the translation
        # was not conducted. A random text is generated then to penalize
        # the translator
        translation_fl = get_random_text(len(source))
    elif translation_tl == translation_fl:
        # if the forward and backward translations are equal, it means that
        # the translation was not conducted. A random text is generated then
        # to penalize the translator
        translation_fl = get_random_text(len(source))
    elif translation_tl == '<translation_missing>' or \
         translation_fl == '<translation_missing>':
        # if the translation is missing, we penalize the translator
        translation_fl = get_random_text(len(source))
    else:
        # if the language of the translation is not guarani, we assume the 
        # translation was not conducted. A random text is generated then to 
        # penalize the translator
        if lang_forward_trans != 'grn':
            translation_fl = get_random_text(len(source))
    return translation_fl


def run_pair_evaluation(translation_dict, from_lang, to_lang, metrics):
    # clean text
    source = clean_text(translation_dict[f'source_text_{from_lang}'])
    translation_fl = clean_text(translation_dict[f'translated_{from_lang}_text'])
    translation_tl = clean_text(translation_dict[f'translated_{to_lang}_text'])
    # record clean text
    translation_dict[f'source_text_{from_lang}'] = source
    translation_dict[f'translated_{from_lang}_text'] = translation_fl
    translation_dict[f'translated_{to_lang}_text'] = translation_tl
    lang_forward_trans = translation_dict.get(f'translated_{to_lang}_language', '')
    translation_fl = validate_translation(source, translation_tl, translation_fl, lang_forward_trans)
    # tokenize text
    source = ' '.join(tokenize(source))
    translation_fl = ' '.join(tokenize(translation_fl))
    # evaluate metrics
    for metric in metrics:
        eval_result = evaluate_results(translation_fl, [source], metric)
        if 'evaluation' not in translation_dict:
            translation_dict['evaluation'] = {}
        if eval_result:
            translation_dict['evaluation'][metric['name']] = eval_result.score
    return translation_dict, source, translation_fl


def run_evaluation(result_file_path, from_lang, to_lang, metrics, bench_data):
    predictions = []
    references = []
    new_model_results = {}
    bleu_scores = defaultdict(list)
    chrf_scores = defaultdict(list)
    with open(result_file_path, 'r') as f:
        model_results = json.load(f)
    new_model_results['model'] = model_results['model']
    new_model_results['params'] = model_results['params']
    new_model_results['rtt_translation'] = []
    new_model_results['evaluation'] = {}
    translations_dict = model_results['rtt_translation']
    # run evaluation of pair translations
    for idx, translation_dict in enumerate(tqdm(translations_dict, desc='Evaluating pair translations...'), start=1):
        translation_dict, source, translation = run_pair_evaluation(
            translation_dict, from_lang, to_lang, metrics
        )
        if 'id' in translation_dict:
            translation_domain = bench_data[translation_dict['id']]
        else:
            translation_domain = bench_data[idx]
        # add to overall list of references and predictions
        predictions.append(translation)
        references.append(source)
        bleu_scores[translation_domain].append(translation_dict['evaluation']['sacrebleu'])
        chrf_scores[translation_domain].append(translation_dict['evaluation']['chrf++'])
        new_model_results['rtt_translation'].append(translation_dict)
    # run overall evaluation
    print('Conducting overall evaluation...')
    model_name = model_results['model']['name'].split('/')[1] if '/' in model_results['model']['name'] else model_results['model']['name']
    overall_eval = {'model_name': model_name}
    for metric in metrics:
        eval_result = evaluate_results(predictions, references, metric, 'corpus')
        if eval_result:
            new_model_results['evaluation'][metric['name']] = eval_result.score
            overall_eval[metric['name']] = eval_result.score
    # compute the number of actual translations based on the detected language 
    # in the translation output and the expected language code
    expected_target_lang = iso_to_detection_code(to_lang)
    expected_source_lang = iso_to_detection_code(from_lang)
    actual_target_translations = compute_actual_translations(
        translations_dict, f'translated_{to_lang}_language', expected_target_lang
    )
    actual_source_translations = compute_actual_translations(
        translations_dict, f'translated_{from_lang}_language', expected_source_lang
    )
    new_model_results['evaluation'][f'actual_{to_lang}_translations'] = \
        actual_target_translations
    new_model_results['evaluation'][f'actual_{from_lang}_translations'] = \
        actual_source_translations
    overall_eval[f'actual_{to_lang}_translations'] = actual_target_translations
    overall_eval[f'actual_{from_lang}_translations'] = actual_source_translations
    # compute the number of valid translations
    num_valid_target_translations = compute_num_valid_translations(
        translations_dict, to_lang
    )
    num_valid_source_translations = compute_num_valid_translations(
        translations_dict, from_lang
    )
    new_model_results['evaluation'][f'valid_{to_lang}_translations'] = \
        num_valid_target_translations
    new_model_results['evaluation'][f'valid_{from_lang}_translations'] = \
        num_valid_source_translations
    overall_eval[f'valid_{to_lang}_translations'] = num_valid_target_translations
    overall_eval[f'valid_{from_lang}_translations'] = num_valid_source_translations
    # compute the number of language disagreements between the detected language 
    # and the valid translation flag
    num_target_language_disagreement = compute_num_language_disagreement(
        translations_dict, to_lang
    )
    num_source_language_disagreement = compute_num_language_disagreement(
        translations_dict, from_lang
    )
    new_model_results['evaluation'][f'{to_lang}_language_disagreement'] = \
        num_target_language_disagreement
    new_model_results['evaluation'][f'{from_lang}_language_disagreement'] = \
        num_source_language_disagreement
    overall_eval[f'{to_lang}_language_disagreement'] = num_target_language_disagreement
    overall_eval[f'{from_lang}_language_disagreement'] = num_source_language_disagreement
    # compute RTT scores
    rtt_score_domains, rtt_score_general = compute_rtt_score(bleu_scores)
    new_model_results['evaluation']['rtt_sacrebleu'] = rtt_score_general
    new_model_results['evaluation']['rtt_sacrebleu_domains'] = rtt_score_domains
    overall_eval['rtt_sacrebleu'] = rtt_score_general
    rtt_score_domains, rtt_score_general = compute_rtt_score(chrf_scores)
    new_model_results['evaluation']['rtt_chrf++'] = rtt_score_general
    new_model_results['evaluation']['rtt_chrf++_domains'] = rtt_score_domains
    overall_eval['rtt_chrf++'] = rtt_score_general
    # save results
    print('Saving results...')
    with open(result_file_path, 'w') as f:
        json.dump(new_model_results, f, ensure_ascii=False, indent=4)
    return overall_eval
    

def load_metrics(metric_names):
    metrics = []
    for metric_name in metric_names:
        if metric_name == 'sacrebleu':
            metrics.append(
                {'name': 'sacrebleu', 'obj': BLEU(effective_order=True)}
            )
        elif metric_name == 'chrf++':
            metrics.append(
                {'name': 'chrf++', 'obj': CHRF(word_order=2)}
            )
    return metrics


def get_bench_data(project_dir, lang):
    if lang == 'es':
        bench_dataset_file_path = os.path.join(project_dir, 'data', 'RTTBench-Mono-ES.jsonl')
    else:
        bench_dataset_file_path = os.path.join(project_dir, 'data', 'RTTBench-Mono.jsonl')
    data = read_jsonl(bench_dataset_file_path)
    bench_data = {d['id']: d['domain'] for d in data}
    return bench_data
 

@click.command()
@click.option('--res_dir', default='', help='Name of the directory containing the results (only the name, not the full path)')
def main(res_dir):
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    metrics = load_metrics(['sacrebleu', 'chrf++'])
    if not res_dir:
        print('Name of the directory containing the results should be included '\
              'as parameter (only the name)')
        return
    from_lang = res_dir.split('_')[0]
    to_lang = res_dir.split('_')[1]
    bench_data = get_bench_data(project_dir, from_lang)
    results_dir = os.path.join(project_dir, 'outputs', 'rtt_experiment', res_dir)
    result_files = [
        e for e in os.listdir(results_dir) 
        if os.path.isfile(os.path.join(results_dir, e)) and e.endswith('.json')
    ]
    eval_results = []
    for model_result_file in tqdm(result_files, desc='Evaluating model translations...'):
        result_file_path = os.path.join(results_dir, model_result_file)
        model_name = model_result_file.split('_')[0]
        print(f'\n\nEvaluation results of the model: {model_name}')
        model_eval_results = run_evaluation(
            result_file_path, from_lang, to_lang, metrics, bench_data
        )
        eval_results.append(model_eval_results)
    eval_results_df = pd.DataFrame(eval_results)
    eval_results_df.to_csv(os.path.join(results_dir, f'overall_evaluation_{res_dir}.csv'), index=False)
    print('Evaluation has successfully finished.')


if __name__ == '__main__':
    main()