import json

from lorem_text import lorem
from nltk.tokenize import TweetTokenizer


_tokenizer = TweetTokenizer(preserve_case=False)


def clean_text(text: str):
    text = text.replace("`", "").replace("‘","'")
    text = text.replace("‘", "'").replace("’","'")
    text = text.replace("\"","'")
    # remove leading and traling single quotes
    text = text.strip("'")
    # if a period is present remove everything after the period
    # since texts correspond to single sentences that don't have punctuations, 
    # we assume that the text after the period can be removed
    punctuations = ['.', '(', ';']
    for punctuation in punctuations:
        if punctuation in text:
            text = text.split(punctuation)[0]
    return text


def tokenize(text: str):
    """
    Tokenizes text using NLTK TweetTokenizer and yields alphabetic tokens only.
    """
    for token in _tokenizer.tokenize(text):
        if token.isalpha():
            yield token


def get_random_text(num_words: int):
    return lorem.words(num_words)


def read_jsonl(dataset_filepath):
    with open(dataset_filepath, 'r') as f:
        return [json.loads(line) for line in f]


def write_jsonl(output_filepath, data, mode='w'):
    with open(output_filepath, mode, encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')