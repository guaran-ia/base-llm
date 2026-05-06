#!/usr/bin/env python3
import click
import json
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.eval_rtt_exp import evaluate_results, load_metrics, run_pair_evaluation
from src.utils.utils import iso_to_detection_code, read_jsonl


def get_bench_dataset_path(project_dir, lang):
    if lang == 'es':
        return os.path.join(project_dir, 'data', 'RTTBench-Mono-ES.jsonl')
    return os.path.join(project_dir, 'data', 'RTTBench-Mono.jsonl')


def load_bench_data(path):
    data = read_jsonl(path)
    return {item['id']: item['domain'] for item in data}


def compute_model_metrics(result_file_path, from_lang, to_lang, metrics, bench_data):
    with open(result_file_path, 'r', encoding='utf-8') as f:
        model_results = json.load(f)

    predictions = []
    references = []
    sentence_rows = []
    bleu_scores = defaultdict(list)
    chrf_scores = defaultdict(list)

    translations = model_results.get('rtt_translation', [])
    for idx, translation_dict in enumerate(translations, start=1):
        translation_dict_copy = translation_dict.copy()
        translation_dict_copy, source, prediction = run_pair_evaluation(
            translation_dict_copy, from_lang, to_lang, metrics
        )

        if 'id' in translation_dict_copy:
            item_id = translation_dict_copy['id']
        else:
            item_id = idx

        domain = bench_data.get(item_id, 'unknown')
        predictions.append(prediction)
        references.append(source)

        bleu_score = translation_dict_copy['evaluation'].get('sacrebleu')
        chrf_score = translation_dict_copy['evaluation'].get('chrf++')
        bleu_scores[domain].append(bleu_score)
        chrf_scores[domain].append(chrf_score)

        sentence_rows.append({
            'id': item_id,
            'domain': domain,
            'source': source,
            'prediction': prediction,
            'sacrebleu': bleu_score,
            'chrf++': chrf_score,
        })

    corpus_metrics = {}
    for metric in metrics:
        result = evaluate_results(predictions, references, metric, mode='corpus')
        corpus_metrics[metric['name']] = result.score if result else None

    total_translations = len(translations)
    expected_target_lang = iso_to_detection_code(to_lang)
    expected_source_lang = iso_to_detection_code(from_lang)
    actual_target_count = 0
    actual_source_count = 0
    for translation_dict in translations:
        if translation_dict.get(f'translated_{to_lang}_language', '').lower() == expected_target_lang:
            actual_target_count += 1
        if translation_dict.get(f'translated_{from_lang}_language', '').lower() == expected_source_lang:
            actual_source_count += 1
    overall_language_rates = {
        f'actual_{to_lang}_translation_rate': actual_target_count / total_translations if total_translations else 0.0,
        f'actual_{from_lang}_translation_rate': actual_source_count / total_translations if total_translations else 0.0,
    }

    domain_avg_scores = {
        'sacrebleu': {domain: sum(values) / len(values) for domain, values in bleu_scores.items()},
        'chrf++': {domain: sum(values) / len(values) for domain, values in chrf_scores.items()},
    }
    overall_rtt_scores = {
        'rtt_sacrebleu': sum(domain_avg_scores['sacrebleu'].values()) / len(domain_avg_scores['sacrebleu']) if domain_avg_scores['sacrebleu'] else 0.0,
        'rtt_chrf++': sum(domain_avg_scores['chrf++'].values()) / len(domain_avg_scores['chrf++']) if domain_avg_scores['chrf++'] else 0.0,
    }

    return {
        'model_name': model_results.get('model', {}).get('name', '<unknown>'),
        'corpus_metrics': corpus_metrics,
        'domain_avg_scores': domain_avg_scores,
        'overall_language_rates': overall_language_rates,
        'overall_rtt_scores': overall_rtt_scores,
        'sentence_rows': sentence_rows,
        'existing_evaluation': model_results.get('evaluation', {}),
    }


def format_comparison(computed, existing, key):
    existing_value = existing.get(key)
    computed_value = computed.get(key)
    if existing_value is None:
        return f'{key}: computed={computed_value:.4f} (existing missing)'
    return f'{key}: computed={computed_value:.4f}, existing={existing_value:.4f}'


@click.command()
@click.option(
    '--result-file',
    default=os.path.join(PROJECT_DIR, 'outputs', 'rtt_experiment', 'es_gn_20260415165900', 'mistral-7b-instruct-v0.3_rtt_results.json'),
    help='Path to the model JSON result file.',
)
@click.option('--from-lang', default='es', help='Source language code used in the RTT result file.')
@click.option('--to-lang', default='gn', help='Target language code used in the RTT result file.')
@click.option('--bench-file', default=None, help='Optional benchmark dataset JSONL path.')
@click.option('--top', type=int, default=10, help='Number of example rows to display.')
def main(result_file, from_lang, to_lang, bench_file, top):
    bench_file = bench_file or get_bench_dataset_path(PROJECT_DIR, from_lang)

    if not os.path.exists(result_file):
        raise FileNotFoundError(f'Result file not found: {result_file}')
    if not os.path.exists(bench_file):
        raise FileNotFoundError(f'Benchmark dataset not found: {bench_file}')

    bench_data = load_bench_data(bench_file)
    metrics = load_metrics(['sacrebleu', 'chrf++'])
    summary = compute_model_metrics(result_file, from_lang, to_lang, metrics, bench_data)

    print(f"Model: {summary['model_name']}")
    print('Corpus metrics:')
    for name, score in summary['corpus_metrics'].items():
        print(f'  {name}: {score:.4f}')

    print('\nDomain average scores:')
    for metric_name, domain_scores in summary['domain_avg_scores'].items():
        print(f'  {metric_name}:')
        for domain, value in sorted(domain_scores.items()):
            print(f'    {domain}: {value:.4f}')

    print('\nLanguage accuracy rates:')
    for key, value in summary['overall_language_rates'].items():
        print(f'  {key}: {value:.4f}')

    print('\nRTT scores:')
    for key, value in summary['overall_rtt_scores'].items():
        print(f'  {key}: {value:.4f}')

    if summary['existing_evaluation']:
        print('\nComparison with existing saved evaluation values:')
        for key in ['sacrebleu', 'chrf++', 'actual_gn_translation_rate', 'actual_es_translation_rate', 'rtt_sacrebleu', 'rtt_chrf++']:
            source_values = summary['corpus_metrics'] if key in summary['corpus_metrics'] else (summary['overall_language_rates'] if key in summary['overall_language_rates'] else summary['overall_rtt_scores'])
            print('  ' + format_comparison(source_values, summary['existing_evaluation'], key))

    print(f"\nFirst {top} examples:")
    for row in summary['sentence_rows'][: top]:
        print(f"- id={row['id']} domain={row['domain']} sacrebleu={row['sacrebleu']:.4f} chrf++={row['chrf++']:.4f}")
        print(f"  source={row['source']}")
        print(f"  prediction={row['prediction']}\n")


if __name__ == '__main__':
    main()
