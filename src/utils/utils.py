import json

from lorem_text import lorem
from nltk.tokenize import TweetTokenizer


_tokenizer = TweetTokenizer(preserve_case=False)


def clean_text(text: str):
    text = text.replace("`", "").replace("‘","'")
    text = text.replace("‘", "'").replace("’","'")
    text = text.replace("\"","'")
    text = text.strip("'")
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


def read_bench_data(dataset_filepath):
    with open(dataset_filepath, 'r') as f:
        return [json.loads(line) for line in f]