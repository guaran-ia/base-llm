import math
from pathlib import Path
from typing import List, Tuple

from src.utils.utils import read_bench_data,extract_text, sentences, words


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
# Settings
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data" / "coreguapa_identified_gn.jsonl"
MIN_WORDS = 3


# ----------------------------
# Core: lengths + mean + std
# ----------------------------
def all_sentence_lengths(texts: List[str], min_words: int = MIN_WORDS) -> List[int]:
    """
    Compute sentence lengths in words across a list of documents.

    Each document may contain multiple sentences. For each sentence:
    - Split it using `sentences()`
    - Count the number of words using `words()`
    - Keep it only if it has at least `min_words` words

    Args:
        texts (List[str]): List of documents.
        min_words (int): Minimum number of words required for a sentence
            to be counted.

    Returns:
        List[int]: List of sentence lengths in words.
    """
    lengths: List[int] = []

    for text in texts:
        for sentence in sentences(text):
            sentence = sentence.strip()

            if not sentence:
                continue

            n_words = len(words(sentence))

            if n_words >= min_words:
                lengths.append(n_words)

    return lengths


def mean_and_std(lengths: List[int]) -> Tuple[float, float]:
    """
    Compute the mean and standard deviation of sentence lengths.

    Mean:
        mean = sum(x) / N

    Standard deviation:
        std = sqrt(sum((x - mean)^2) / N)

    Args:
        lengths (List[int]): List of sentence lengths in words.

    Returns:
        Tuple[float, float]: Mean and standard deviation. Returns
            (0.0, 0.0) if the input list is empty.
    """
    if not lengths:
        return 0.0, 0.0

    mean_val = sum(lengths) / len(lengths)
    variance = sum((x - mean_val) ** 2 for x in lengths) / len(lengths)
    std_val = math.sqrt(variance)

    return mean_val, std_val


def main() -> None:
    """
    Compute sentence-length mean and standard deviation for a JSONL corpus.

    Returns:
        None
    """
    rows = read_bench_data(INPUT_FILE)
    texts = list(extract_text(rows, field="text"))

    lengths = all_sentence_lengths(texts, min_words=MIN_WORDS)
    mean_val, std_val = mean_and_std(lengths)

    print(f"Total sentences: {len(lengths)}")
    print(f"Mean words per sentence: {mean_val:.2f}")
    print(f"Std words per sentence: {std_val:.2f}")


if __name__ == "__main__":
    main()