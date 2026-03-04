import json
from pathlib import Path
from typing import Iterator, List, Tuple

import spacy
from nltk.tokenize import TweetTokenizer


"""
    SETUP REQUIREMENT:

    This script depends on the CoreGuapa corpus file:
        data/coreguapa_identified_gn.jsonl

    You must manually copy this file into the `data/` directory
    before executing the script.

    The corpus is used to estimate sentence-length distribution
    (mean and std) for defining generation constraints.
"""

# ----------------------------
# Sentence segmentation
# ----------------------------
_nlp = spacy.blank("xx")
_nlp.add_pipe("sentencizer")
_nlp.max_length = 10_000_000


def sentences(text: str) -> List[str]:
    """
    Split a text into sentences using spaCy's sentencizer.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of sentence strings.
    """
    doc = _nlp(text or "")
    return [sent.text for sent in doc.sents]


# ----------------------------
# Tokenization
# ----------------------------
_tokenizer = TweetTokenizer(preserve_case=False)


def tokenize(text: str) -> Iterator[str]:
    """
    Tokenize a text using NLTK TweetTokenizer and yield alphabetic tokens only.

    Args:
        text (str): Input text.

    Yields:
        str: Lowercased alphabetic tokens (non-alphabetic tokens are skipped).
    """
    for token in _tokenizer.tokenize(text or ""):
        if token.isalpha():
            yield token


def words(text: str) -> List[str]:
    """
    Convert a text into a list of tokens using `tokenize()`.

    Args:
        text (str): Input text.

    Returns:
        List[str]: List of alphabetic tokens.
    """
    return list(tokenize(text))


# ----------------------------
# JSONL Reader
# ----------------------------
def read_jsonl(path: Path) -> Iterator[dict]:
    """
    Read a JSONL file and yield one JSON object per line.

    Args:
        path (Path): Path to the JSONL file.

    Yields:
        dict: Parsed JSON object from each non-empty line.

    Raises:
        json.JSONDecodeError: If a line contains invalid JSON.
        FileNotFoundError: If `path` does not exist.
    """
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_text(record: dict) -> str:
    """
    Extract text content from a JSON record using common field names.

    The function checks, in order: "text", "sentence", "content".

    Args:
        record (dict): JSON record.

    Returns:
        str: Extracted text if present, otherwise an empty string.
    """
    for key in ("text", "sentence", "content"):
        value = record.get(key)
        if isinstance(value, str):
            return value
    return ""


# ----------------------------
# Core: lengths + mean + std
# ----------------------------
def all_sentence_lengths(texts: List[str], min_words: int = 3) -> List[int]:
    """
    Compute sentence lengths (in words) across a list of documents.

    Each document may contain multiple sentences. For each sentence:
    - Tokenize it using `words()`
    - Count the number of tokens (words)
    - Keep it only if it has at least `min_words` words

    Args:
        texts (List[str]): List of documents (each string may contain multiple sentences).
        min_words (int): Minimum number of words required for a sentence to be counted.

    Returns:
        List[int]: List of sentence lengths (word counts).
    """
    lengths: List[int] = []

    for text in texts:
        for s in sentences(text):
            s = s.strip()
            if not s:
                continue

            n = len(words(s))
            if n >= min_words:
                lengths.append(n)

    return lengths


from typing import List, Tuple
import math

def mean_and_std(lengths: List[int]) -> Tuple[float, float]:
    """
    Compute mean and standard deviation from a list of lengths.

    Mean:
        mean = sum(x) / N

    Standard Deviation:
        std = sqrt(sum((x - mean)^2) / N)

    Args:
        lengths (List[int]): List of sentence lengths (word counts).

    Returns:
        Tuple[float, float]: (mean, std). Returns (0.0, 0.0) if the input is empty.
    """
    if not lengths:
        return 0.0, 0.0

    mean_val = sum(lengths) / len(lengths)

    variance = sum((x - mean_val) ** 2 for x in lengths) / len(lengths)
    std_val = math.sqrt(variance)

    return mean_val, std_val


if __name__ == "__main__":
    """
    Compute sentence-length mean and Std for a JSONL corpus.

    """
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    dataset_path = PROJECT_ROOT / "data" / "coreguapa_identified_gn.jsonl"

    texts = [extract_text(record) for record in read_jsonl(dataset_path)]

    lengths = all_sentence_lengths(texts, min_words=3)
    mean_val, std_val = mean_and_std(lengths)

    print(f"Total sentences: {len(lengths)}")
    print(f"Mean words per sentence: {mean_val:.2f}")
    print(f"Std words per sentence: {std_val:.2f}")