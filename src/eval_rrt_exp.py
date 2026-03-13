import evaluate

from nltk.tokenize import TweetTokenizer


_tokenizer = TweetTokenizer(preserve_case=False)


def tokenize(text: str):
    """
    Tokenizes text using NLTK TweetTokenizer and yields alphabetic tokens only.
    """
    for token in _tokenizer.tokenize(text):
        if token.isalpha():
            yield token


def evaluate_results(trans_results, metric_name):
    predictions = [tokenize(result['translated_es_text']) for result in trans_results['rtt_translation']]
    references = [[tokenize(result['source'])] for result in trans_results['rtt_translation']]
    if metric_name == 'sacrebleu':
        metric = evaluate.load('sacrebleu')
        eval_results = metric.compute(predictions=predictions, references=references)
    elif metric_name == 'chrf++':
        metric = evaluate.load('chrf')
        eval_results = metric.compute(
            predictions=predictions, 
            references=references,
            word_order=2
        )
    else:
        print(f'Unknown metric: {metric_name}')
    
    if eval_results:
        trans_results['evaluation'] = {metric_name: eval_results}
    return eval_results


def main():
    rtt_model = []
    for metric in ['sacrebleu', 'chrf++']:
        print(f'Evaluating results using {metric} metric...')
        rtt_model = evaluate_results(rtt_model, metric)