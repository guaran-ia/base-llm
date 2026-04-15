# Round-Trip Translation Experiments: The Case of Guarani

This repository contains the scripts, data, and evaluation code for 
benchmarking open-weight base LLMs on their ability to communicate in Guarani.

The experiment uses a round-trip translation (RTT) strategy: Spanish sentences 
are translated to Guarani, then translated back to Spanish. Results are evaluated 
by comparing the original Spanish sentence with the back-translated Spanish sentence.

## Dataset

The experiments are conducted on a synthetic Spanish dataset of 1,250 sentences 
across 25 domains such as Arts, Literature, Business, and Technology.

Key datasets:
- `data/RTTBench-Mono-ES.jsonl` — 1,250 synthetic Spanish sentences used for Spanish→Guarani→Spanish RTT.
- `data/RTTBench-Mono.jsonl` — 1,250 synthetic English sentences used for English→Guarani→English RTT.

## Models tested

Model configurations are defined in `data/rtt_experiments/es_gn/base_models.json`.

Tested models include:
- Gemma-4: `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`
- Gemma-3: `google/gemma-3-4b-it`, `google/gemma-3-12b-it`
- Apertus: `swiss-ai/Apertus-8B-Instruct-2509`
- Qwen3: `Qwen/Qwen3-4B-Instruct-2507`
- Qwen3.5: `Qwen/Qwen3.5-9B`
- Meta-Llama-3.1: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Nemotron-Mini-4B-Instruct: `nvidia/Nemotron-Mini-4B-Instruct`
- Mistral-NeMo-Minitron-8B-Instruct: `nvidia/Mistral-NeMo-Minitron-8B-Instruct`
- Mistral-7B-Instruct-v0.3: `mistralai/Mistral-7B-Instruct-v0.3`
- Grok: `grok-4-fast-non-reasoning`
- GPT: `gpt-4o-mini`

## Experiment design

The RTT workflow is:
1. Translate Spanish source sentences to Guarani.
2. Translate generated Guarani translations back to Spanish.
3. Compare original Spanish sentences with back-translated Spanish sentences.

This design measures how well models preserve meaning and fluency across the Spanish↔Guarani translation round trip.

## Evaluation metrics

Evaluation is based on commonly used translation quality metrics and RTT-specific scoring:

- BLEU
- chrF++
- RTTScore (from Zamir et al., [https://arxiv.org/pdf/2601.10804](https://arxiv.org/pdf/2601.10804))

RTTScore is used to enable domain-conditioned evaluation and to better understand how models generalize across domains.

## Repository structure

- `src/` — scripts and utility code used for dataset preparation, experiment execution, and evaluation.
- `data/` — input datasets and experiment configuration files.
- `outputs/` — generated model outputs, translations, and evaluation reports.
- `data/rtt_experiments/es_gn/base_models.json` — model list for the Guarani evaluation.

## Getting started

1. Clone the repository.
2. Create and activate a Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the available scripts under `src/` to prepare datasets, submit batch jobs, and evaluate outputs.

## Dependencies

Primary dependencies are listed in `requirements.txt`, including:
- `click`
- `fasttext`
- `huggingface_hub`
- `nltk`
- `numpy`
- `openai`
- `pandas`
- `python-dotenv`
- `sacrebleu`
- `spacy`
- `torch`
- `tqdm`
- `transformers`
- `lorem-text`
- `matplotlib`
- `sentence-transformers`

## Notes

- The Spanish dataset was generated with Azure OpenAI GPT-4.1.
- The experiments aim to reveal how translation quality varies across domains and 
how well models can generalize the Spanish↔Guarani RTT task.
- Outputs and evaluation reports are saved under `outputs/`.
