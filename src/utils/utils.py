import json
from pathlib import Path
from typing import Any, Dict, Iterator, List

import spacy
from lorem_text import lorem
from nltk.tokenize import TweetTokenizer
# ---------------------------------------------------------------------
# NLP initialization for sentence segmentation
# ---------------------------------------------------------------------

_nlp = spacy.blank("xx")
_nlp.add_pipe("sentencizer")
_nlp.max_length = 10_000_000


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


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Write a list of dictionaries to a JSONL file.

    Args:
        path (Path): Output JSONL file path.
        rows (List[Dict[str, Any]]): Rows to write.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: Path) -> Any:
    """
    Read a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        Any: Parsed JSON content.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    """
    Write data to a JSON file using pretty formatting.

    Args:
        path (Path): Output JSON file path.
        data (Any): JSON-serializable object.

    Returns:
        None
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def sentences(text: str) -> List[str]:
    """
    Split a text into sentences using spaCy sentencizer.

    Args:
        text (str): Input text.

    Returns:
        List[str]: A list of cleaned sentences.
    """
    if not text or not text.strip():
        return []

    doc = _nlp(text)

    return [
        sent.text.strip()
        for sent in doc.sents
        if sent.text.strip()
    ]


def words(text: str) -> List[str]:
    """
    Return word-like tokens, filtering out tokens without alphanumeric characters.
    """
    return list(tokenize(text))


def extract_text(rows: List[dict], field: str = "text") -> Iterator[str]:
    """
    Yield the text field from each row if present and non-empty.

    Args:
        rows (List[dict]): Input rows.
        field (str): Field name containing the text.

    Yields:
        str: Cleaned text values.
    """
    for row in rows:
        value = row.get(field, "")

        if isinstance(value, str) and value.strip():
            yield value.strip()


def sentence_word_counts(text: str) -> List[int]:
    """
    Compute the number of words in each sentence of a text.

    Args:
        text (str): Input text.

    Returns:
        List[int]: Word counts per sentence.
    """
    return [
        len(words(sentence))
        for sentence in sentences(text)
    ]


def flatten_sentences(rows: List[dict], field: str = "text") -> List[str]:
    """
    Extract and flatten all sentences from a collection of rows.

    Args:
        rows (List[dict]): Input rows.
        field (str): Field name containing the text.

    Returns:
        List[str]: A flat list containing all sentences.
    """
    all_sentences: List[str] = []

    for text in extract_text(rows, field=field):
        all_sentences.extend(sentences(text))

    return all_sentences
