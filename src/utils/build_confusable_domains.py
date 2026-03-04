import json
from pathlib import Path
from typing import Dict, List, Any, Tuple

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download


# -------------------------------
# Settings
# -------------------------------
MODEL_ID = "nvidia/domain-classifier"

BASE_DIR    = Path(__file__).resolve().parent.parent
INPUT_FILE  = BASE_DIR / "data" / "dataset.jsonl"   
OUTPUT_FILE = BASE_DIR / "data" / "domains.json"    # JSON array with confusables

TOP_K = 3
BATCH_SIZE = 32
MAX_LEN = 256


# -------------------------------
# Helpers
# -------------------------------
def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_hub_json_config(model_id: str) -> Dict[str, Any]:
    # The hub config.json is what CustomModel expects (base_model, fc_dropout, id2label, etc.)
    cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
    return json.loads(Path(cfg_path).read_text(encoding="utf-8"))


# -------------------------------
# Model definition (robust dtype)
# -------------------------------
class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(float(config.get("fc_dropout", 0.0)))
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

        raw = config["id2label"]
        self.id2label = {int(k): v for k, v in raw.items()} if isinstance(raw, dict) else raw
        self.label2id = {v: k for k, v in self.id2label.items()}

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)

        # ---- FIX: match dtype with Linear weights (prevents Half vs Float crash)
        dropped = dropped.to(self.fc.weight.dtype)

        logits = self.fc(dropped)
        return torch.softmax(logits[:, 0, :], dim=1)  # (batch, num_labels)


# -------------------------------
# Core logic: compute confusables
# -------------------------------
@torch.inference_mode()
def compute_confusables(rows: List[dict]) -> List[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hub_cfg = load_hub_json_config(MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = CustomModel.from_pretrained(MODEL_ID).to(device)
    model = model.float()  # ---- FIX: force everything to fp32 (stable)
    model.eval()

    num_labels = len(model.id2label)

    # Accumulators: sums[domain] = sum prob vectors, counts[domain] = num samples
    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]

        texts = [r["text"] for r in batch]
        true_domains = [r["domain"] for r in batch]

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        probs = model(inputs["input_ids"], inputs["attention_mask"])  # (B, num_labels)

        for j, true_domain in enumerate(true_domains):
            if true_domain not in sums:
                sums[true_domain] = torch.zeros(num_labels, dtype=torch.float32, device="cpu")
                counts[true_domain] = 0

            sums[true_domain] += probs[j].detach().to("cpu")
            counts[true_domain] += 1

    results: List[dict] = []

    for domain_name, prob_sum in sums.items():
        n = counts[domain_name]
        avg = prob_sum / max(n, 1)
        avg_list = avg.tolist()

        # Zero-out self if label exists in classifier label set
        if domain_name in model.label2id:
            avg_list[model.label2id[domain_name]] = 0.0

        ranked = sorted(
            [(model.id2label[idx], float(p)) for idx, p in enumerate(avg_list)],
            key=lambda x: x[1],
            reverse=True,
        )

        confusables: List[str] = []
        for label, p in ranked:
            confusables.append(label)
            if len(confusables) >= TOP_K:
                break

        results.append({
            "name": domain_name,
            "description": "",
            "confusables": confusables,
        })

    results.sort(key=lambda x: x["name"].lower())
    return results


def main():
    rows = read_jsonl(INPUT_FILE)
    if not rows:
        raise ValueError(f"No rows found in {INPUT_FILE}")

    bad = [r for r in rows if "domain" not in r or "text" not in r]
    if bad:
        raise ValueError("Some rows are missing 'domain' or 'text'. Example: " + str(bad[0]))

    results = compute_confusables(rows)

    OUTPUT_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Wrote {len(results)} domains to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()