import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


DEFAULT_MODEL_ID = "nvidia/multilingual-domain-classifier"


def validate_domains(domains: List[dict]) -> None:
    """
    Validate that domain definitions contain required fields.

    Args:
        domains (List[dict]): Domain definition rows.

    Returns:
        None

    Raises:
        ValueError: If domain definitions are missing or invalid.
    """
    if not domains:
        raise ValueError("No domains found.")

    invalid_domains = [
        domain
        for domain in domains
        if "name" not in domain
        or not isinstance(domain["name"], dict)
        or "es" not in domain["name"]
        or "en" not in domain["name"]
    ]

    if invalid_domains:
        raise ValueError(
            "Some domain definitions are invalid. "
            f"Example: {invalid_domains[0]}"
        )


def build_domain_maps(domains: List[dict]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Build mappings between English and Spanish domain names.

    Args:
        domains (List[dict]): Domain definition rows.

    Returns:
        Tuple[Dict[str, str], Dict[str, str]]:
            en_to_es, es_to_en
    """
    en_to_es: Dict[str, str] = {}
    es_to_en: Dict[str, str] = {}

    for domain in domains:
        es_name = domain["name"]["es"]
        en_name = domain["name"]["en"]

        en_to_es[en_name] = es_name
        es_to_en[es_name] = en_name

    return en_to_es, es_to_en


def build_domain_metadata(domains: List[dict]) -> Dict[str, dict]:
    """
    Index domain metadata by Spanish name.

    Args:
        domains (List[dict]): Domain definition rows.

    Returns:
        Dict[str, dict]: Metadata indexed by Spanish domain.
    """
    metadata_by_es: Dict[str, dict] = {}

    for domain in domains:
        metadata_by_es[domain["name"]["es"]] = domain

    return metadata_by_es


def map_domain_to_english(domain_name: str, es_to_en: Dict[str, str]) -> str:
    """
    Map a Spanish domain label to its English equivalent.

    Args:
        domain_name (str): Domain name from dataset.
        es_to_en (Dict[str, str]): Spanish -> English mapping.

    Returns:
        str: English domain label if found, otherwise the original value.
    """
    return es_to_en.get(domain_name, domain_name)


def load_hub_json_config(model_id: str) -> Dict[str, Any]:
    """
    Download and load the model config.json from Hugging Face Hub.

    Args:
        model_id (str): Hugging Face model id.

    Returns:
        Dict[str, Any]: Parsed JSON config.
    """
    cfg_path = hf_hub_download(repo_id=model_id, filename="config.json")
    return json.loads(Path(cfg_path).read_text(encoding="utf-8"))


class CustomModel(nn.Module, PyTorchModelHubMixin):
    """
    Wrapper around the NVIDIA multilingual domain classifier head.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(float(config.get("fc_dropout", 0.0)))

        self.fc = nn.Linear(
            self.model.config.hidden_size,
            len(config["id2label"]),
        )

        raw = config["id2label"]
        self.id2label = (
            {int(k): v for k, v in raw.items()}
            if isinstance(raw, dict)
            else raw
        )
        self.label2id = {v: k for k, v in self.id2label.items()}

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass that returns probabilities.

        Args:
            input_ids (torch.Tensor): Token ids.
            attention_mask (torch.Tensor): Attention mask.

        Returns:
            torch.Tensor: Class probabilities.
        """
        features = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        dropped = self.dropout(features)
        dropped = dropped.to(self.fc.weight.dtype)

        logits = self.fc(dropped)
        probs = torch.softmax(logits[:, 0, :], dim=1)

        return probs


class DomainClassifier:
    """
    Domain classifier utility using NVIDIA multilingual-domain-classifier.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        batch_size: int = 32,
        max_len: int = 256,
    ):
        """
        Initialize classifier.

        Args:
            model_id (str): Model id.
            batch_size (int): Batch size for inference.
            max_len (int): Maximum token length.
        """
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_len = max_len

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = AutoConfig.from_pretrained(model_id)
        self.hub_config = load_hub_json_config(model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = CustomModel.from_pretrained(model_id).to(self.device)

        self.model = self.model.float()
        self.model.eval()

        self.id2label = self.model.id2label
        self.label2id = self.model.label2id

    @torch.inference_mode()
    def predict_probs(self, texts: List[str]) -> torch.Tensor:
        """
        Predict probabilities for a list of texts.

        Args:
            texts (List[str]): Input texts.

        Returns:
            torch.Tensor: Probability tensor of shape [n_texts, n_labels].
        """
        probs_all = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_len,
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            probs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

            probs_all.append(probs.detach().cpu())

        return torch.cat(probs_all, dim=0)

    def predict_topk(
        self,
        texts: List[str],
        k: int = 3,
    ) -> List[List[Tuple[str, float]]]:
        """
        Predict top-k domain labels for each text.

        Args:
            texts (List[str]): Input texts.
            k (int): Number of top labels to return.

        Returns:
            List[List[Tuple[str, float]]]: Top-k label-score pairs per text.
        """
        probs = self.predict_probs(texts)
        top_probs, top_ids = torch.topk(probs, k=k, dim=1)

        results: List[List[Tuple[str, float]]] = []

        for prob_row, id_row in zip(top_probs, top_ids):
            labels = [self.id2label[int(i)] for i in id_row.tolist()]
            scores = [float(s) for s in prob_row.tolist()]
            results.append(list(zip(labels, scores)))

        return results