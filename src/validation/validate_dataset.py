from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

from src.utils.domain_classifier import (
    DEFAULT_MODEL_ID,
    DomainClassifier,
    build_domain_maps,
    map_domain_to_english,
    validate_domains,
)
from src.utils.io import read_json, read_jsonl, write_json, write_jsonl


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

INPUT_FILE = BASE_DIR / "data" / "RTTBench-Mono-ES.jsonl"
DOMAINS_FILE = BASE_DIR / "data" / "domains.json"

REPORT_DIR = BASE_DIR / "data" / "report"

REPORT_FILE = REPORT_DIR / "validation_report.json"
ERRORS_FILE = REPORT_DIR / "validation_errors.jsonl"
TSNE_CSV_FILE = REPORT_DIR / "tsne_coordinates.csv"
TSNE_PLOT_FILE = REPORT_DIR / "tsne_plot.png"

MODEL_ID = DEFAULT_MODEL_ID
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def validate_rows(rows: List[dict]) -> None:
    """
    Validate that input rows contain the required fields.

    Args:
        rows (List[dict]): Dataset rows.

    Returns:
        None

    Raises:
        ValueError: If the dataset is empty or malformed.
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


def validate_with_classifier(rows: List[dict]) -> Dict[str, object]:
    """
    Validate the dataset using the multilingual domain classifier.

    For each row:
    - Map the expected domain from Spanish to English
    - Predict the top-1 domain
    - Compare prediction against the expected label

    Args:
        rows (List[dict]): Dataset rows.

    Returns:
        Dict[str, object]: Overall accuracy, per-domain accuracy,
            and misclassified examples.
    """
    classifier = DomainClassifier(model_id=MODEL_ID)

    texts = [row["text"] for row in rows]

    domains = read_json(DOMAINS_FILE)
    validate_domains(domains)
    en_to_es, es_to_en = build_domain_maps(domains)

    expected_domains_en = [
        map_domain_to_english(row["domain"], es_to_en)
        for row in rows
    ]

    top1_predictions = classifier.predict_topk(texts, k=1)

    total = len(rows)
    top1_correct = 0

    per_domain_total: Dict[str, int] = {}
    per_domain_top1: Dict[str, int] = {}

    errors: List[dict] = []

    for row, expected_domain_en, prediction_list in zip(
        rows,
        expected_domains_en,
        top1_predictions,
    ):
        predicted_label_en, predicted_score = prediction_list[0]
        predicted_label_es = en_to_es.get(predicted_label_en, predicted_label_en)

        expected_domain_es = row["domain"]
        is_top1_correct = predicted_label_en == expected_domain_en
        top1_correct += int(is_top1_correct)

        per_domain_total[expected_domain_es] = (
            per_domain_total.get(expected_domain_es, 0) + 1
        )
        per_domain_top1[expected_domain_es] = (
            per_domain_top1.get(expected_domain_es, 0) + int(is_top1_correct)
        )

        if not is_top1_correct:
            errors.append(
                {
                    "id": row.get("id"),
                    "domain": expected_domain_es,
                    "text": row["text"],
                    "domain_top1": predicted_label_es,
                    "domain_top1_score": float(predicted_score),
                    "top1_correct": is_top1_correct,
                }
            )

    per_domain = {}

    for domain_name_es, count in sorted(per_domain_total.items()):
        per_domain[domain_name_es] = {
            "count": count,
            "top1_accuracy": per_domain_top1[domain_name_es] / count,
        }

    return {
        "n_samples": total,
        "top1_accuracy": top1_correct / total if total else 0.0,
        "per_domain": per_domain,
        "n_errors": len(errors),
        "errors": errors,
    }


def compute_tsne_coordinates(rows: List[dict]) -> pd.DataFrame:
    """
    Compute sentence embeddings and project them to 2D using t-SNE.

    Args:
        rows (List[dict]): Dataset rows.

    Returns:
        pd.DataFrame: DataFrame containing id, domain, text,
            and 2D t-SNE coordinates.
    """
    texts = [row["text"] for row in rows]
    domains = [row["domain"] for row in rows]
    ids = [row.get("id") for row in rows]

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    perplexity = min(30, max(5, len(rows) - 1))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        init="random",
        learning_rate="auto",
    )

    coordinates = tsne.fit_transform(embeddings)

    return pd.DataFrame(
        {
            "id": ids,
            "domain": domains,
            "text": texts,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    )


def save_tsne_plot(df: pd.DataFrame, output_path: Path) -> None:
    """
    Save the t-SNE scatter plot to disk.

    Args:
        df (pd.DataFrame): DataFrame with t-SNE coordinates.
        output_path (Path): Output image path.

    Returns:
        None
    """
    plt.figure(figsize=(16, 12))

    domains = sorted(df["domain"].unique())
    cmap = matplotlib.colormaps.get_cmap("hsv").resampled(len(domains))

    for i, domain_name in enumerate(domains):
        subset = df[df["domain"] == domain_name]

        plt.scatter(
            subset["x"],
            subset["y"],
            color=cmap(i),
            label=domain_name,
            alpha=0.6,
            s=25,
        )

    plt.title("t-SNE visualization of RTTBench-Mono-ES by domain")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def compute_cluster_summary(df: pd.DataFrame) -> Dict[str, object]:
    """
    Compute centroid-based cluster statistics for each domain.

    For each domain:
    - Compute the centroid in 2D t-SNE space
    - Compute the average distance from points to the centroid

    Args:
        df (pd.DataFrame): DataFrame with t-SNE coordinates.

    Returns:
        Dict[str, object]: Cluster summary indexed by domain name.
    """
    summary: Dict[str, object] = {}

    for domain_name in sorted(df["domain"].unique()):
        subset = df[df["domain"] == domain_name].copy()

        centroid_x = float(subset["x"].mean())
        centroid_y = float(subset["y"].mean())

        distances = (
            (subset["x"] - centroid_x) ** 2 +
            (subset["y"] - centroid_y) ** 2
        ).pow(0.5)

        summary[domain_name] = {
            "count": int(len(subset)),
            "centroid": {
                "x": centroid_x,
                "y": centroid_y,
            },
            "avg_distance_to_centroid": float(distances.mean()),
        }

    return summary


def main() -> None:
    """
    Run the full dataset validation pipeline.

    This includes:
    1. Classifier-based validation
    2. Embedding-based validation with t-SNE

    Returns:
        None
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(INPUT_FILE)
    validate_rows(rows)

    # Run classifier-based validation.
    classifier_report = validate_with_classifier(rows)
    write_jsonl(ERRORS_FILE, classifier_report["errors"])

    # Run embedding-based validation with t-SNE.
    tsne_df = compute_tsne_coordinates(rows)
    tsne_df.to_csv(TSNE_CSV_FILE, index=False, encoding="utf-8")
    save_tsne_plot(tsne_df, TSNE_PLOT_FILE)

    cluster_summary = compute_cluster_summary(tsne_df)

    final_report = {
        "input_file": str(INPUT_FILE),
        "classifier_validation": {
            "model_id": MODEL_ID,
            "n_samples": classifier_report["n_samples"],
            "top1_accuracy": classifier_report["top1_accuracy"],
            "per_domain": classifier_report["per_domain"],
            "n_errors": classifier_report["n_errors"],
            "errors_file": str(ERRORS_FILE),
        },
        "embedding_validation": {
            "embedding_model": EMBEDDING_MODEL,
            "n_points": int(len(tsne_df)),
            "tsne_csv_file": str(TSNE_CSV_FILE),
            "tsne_plot_file": str(TSNE_PLOT_FILE),
            "cluster_summary": cluster_summary,
        },
    }

    write_json(REPORT_FILE, final_report)

    print(f"Report directory: {REPORT_DIR}")
    print(f"Validation report: {REPORT_FILE}")
    print(f"Classifier errors: {ERRORS_FILE}")
    print(f"t-SNE coordinates: {TSNE_CSV_FILE}")
    print(f"t-SNE plot: {TSNE_PLOT_FILE}")
    print(f"Top-1 accuracy: {classifier_report['top1_accuracy']:.4f}")


if __name__ == "__main__":
    main()