from typing import Iterator, List

import spacy
from nltk.tokenize import TweetTokenizer


# ---------------------------------------------------------------------
# NLP initialization for sentence segmentation
# ---------------------------------------------------------------------

_nlp = spacy.blank("xx")
_nlp.add_pipe("sentencizer")
_nlp.max_length = 10_000_000

_tokenizer = TweetTokenizer()


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


def tokenize(text: str) -> List[str]:
    """
    Tokenize text using NLTK TweetTokenizer.
    """
    if not text or not text.strip():
        return []

    return _tokenizer.tokenize(text)


def words(text: str) -> List[str]:
    """
    Return word-like tokens, filtering out tokens without alphanumeric characters.
    """
    return [tok for tok in tokenize(text) if any(ch.isalnum() for ch in tok)]


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