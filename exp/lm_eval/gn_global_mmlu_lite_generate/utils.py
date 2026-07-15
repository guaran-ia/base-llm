import re
from typing import Any, Optional


VALID_ANSWERS = {"A", "B", "C", "D"}


def normalize_answer(value: Any) -> Optional[str]:
    if value is None:
        return None
    answer = str(value).strip().upper()
    return answer if answer in VALID_ANSWERS else None


def extract_answer(text: Any) -> Optional[str]:
    """Extract an A/B/C/D answer from a generated response."""
    if isinstance(text, list):
        text = text[0] if text else ""
    if text is None:
        return None

    response = str(text).strip()
    if not response:
        return None

    answer_word = r"(?:answer|respuesta|resposta|mbohov[aá]i|option|op[cç]i[oó]n)"
    correctness_word = r"(?:correct|correcta|correta|hekopete)"
    connector = r"(?:is|es|e|é|ha'?e|ha’e|:|=|-)"
    patterns = [
        rf"(?i)\b(?:a|la|el|the)?\s*{answer_word}\b\s*(?:{correctness_word})?\s*(?:{connector})\s*[\(\[]?\s*([ABCD])\b",
        rf"(?i)\b(?:the\s+correct\s+answer|la\s+respuesta\s+correcta|a\s+resposta\s+correta)\s*(?:{connector})\s*[\(\[]?\s*([ABCD])\b",
        r"^\s*[\(\[]?\s*([ABCD])\s*[\)\]]?\s*[\].,:;]?\s*$",
        r"^\s*[\(\[]?\s*([ABCD])\s*[\)\].,:;]\s+",
    ]
    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            return match.group(1).upper()
    return None


def process_results(doc: dict, results: list[str]) -> dict[str, float]:
    generated = results[0] if results else ""
    prediction = extract_answer(generated)
    gold = normalize_answer(doc.get("answer"))
    return {"acc": 1.0 if prediction is not None and prediction == gold else 0.0}
