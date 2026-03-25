from pathlib import Path
from typing import Dict, List

from src.utils.domain_classifier import (
    DEFAULT_MODEL_ID,
    DomainClassifier,
    build_domain_maps,
    build_domain_metadata,
    validate_domains,
)
from src.utils.utils import read_json, read_jsonl, write_json


# -------------------------------
# Settings
# -------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_FILE = BASE_DIR / "data" / "RTTBench-Mono-ES.jsonl"
DOMAINS_FILE = BASE_DIR / "data" / "domains.json"
OUTPUT_FILE = BASE_DIR / "data" / "domains.json"

TOP_K = 5
MODEL_ID = DEFAULT_MODEL_ID


def validate_rows(rows: List[dict]) -> None:
    """
    Validate input dataset rows.

    Args:
        rows (List[dict]): Dataset rows.

    Returns:
        None

    Raises:
        ValueError: If rows are missing or malformed.
    """
    if not rows:
        raise ValueError(f"No rows found in {INPUT_FILE}")

    invalid_rows = [
        row
        for row in rows
        if "domain" not in row or "text" not in row
    ]

    if invalid_rows:
        raise ValueError(
            "Some rows are missing 'domain' or 'text'. "
            f"Example: {invalid_rows[0]}"
        )


def compute_confusables(
    rows: List[dict],
    domains: List[dict],
    top_k: int = TOP_K,
) -> List[dict]:
    """
    Compute top-k confusable domains for each domain using one
    representative sentence per domain.

    Args:
        rows (List[dict]): Dataset rows.
        domains (List[dict]): Domain definitions.
        top_k (int): Number of confusable domains to keep.

    Returns:
        List[dict]: Updated domain definitions with confusables.
    """
    classifier = DomainClassifier(model_id=MODEL_ID)

    en_to_es, _ = build_domain_maps(domains)
    metadata_by_es = build_domain_metadata(domains)

    selected_rows: Dict[str, dict] = {}

    for row in rows:
        domain_name_es = row["domain"]

        if domain_name_es not in selected_rows:
            selected_rows[domain_name_es] = row

    texts = [row["text"] for row in selected_rows.values()]
    true_domains_es = [row["domain"] for row in selected_rows.values()]

    probs = classifier.predict_probs(texts)

    results: List[dict] = []

    for domain_name_es, prob_vector in zip(true_domains_es, probs):
        scores = prob_vector.tolist()

        ranked = sorted(
            [
                (classifier.id2label[idx], float(score))
                for idx, score in enumerate(scores)
            ],
            key=lambda x: x[1],
            reverse=True,
        )

        confusables_es: List[str] = []

        for label_en, _ in ranked:
            label_es = en_to_es.get(label_en)

            if not label_es:
                continue

            if label_es == domain_name_es:
                continue

            confusables_es.append(label_es)

            if len(confusables_es) >= top_k:
                break

        domain_metadata = metadata_by_es[domain_name_es]

        results.append(
            {
                "name": domain_metadata["name"],
                "description": domain_metadata.get("description", ""),
                "confusables": confusables_es,
            }
        )

    results.sort(key=lambda row: row["name"]["es"].lower())

    return results


def main() -> None:
    """
    Build confusable domains file from dataset and domain definitions.

    Returns:
        None
    """
    rows = read_jsonl(INPUT_FILE)
    validate_rows(rows)

    domains = read_json(DOMAINS_FILE)
    validate_domains(domains)

    results = compute_confusables(rows, domains, top_k=TOP_K)

    write_json(OUTPUT_FILE, results)

    print(f"Wrote {len(results)} domains to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()