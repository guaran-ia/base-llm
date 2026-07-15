# Guarani LLM Evaluation

This repository contains scripts, data, task definitions, and evaluation code for
benchmarking open-weight and commercial LLMs on Guarani-language capabilities.
It supports both translation-oriented and question-answering evaluations.

The current evaluation workflows are:
- **Round-trip translation (RTT):** Spanish or English source sentences are
  translated to Guarani and then translated back to the source language. Results
  are evaluated with lexical, character-level, domain-level, and semantic
  similarity metrics.
- **Global MMLU-Lite in Guarani:** models answer multiple-choice questions from a
  Guarani Global MMLU-Lite dataset. The repository supports both a plain
  generation-based runner and `lm-evaluation-harness` task variants.
- **`lm-eval` task development:** task configs are included for likelihood-based
  multiple-choice scoring and generative answer parsing.

## Datasets

The RTT experiments are conducted on synthetic source-language datasets across
25 domains such as Arts, Literature, Business, and Technology. Global MMLU-Lite
experiments use a Guarani multiple-choice question-answering dataset.

Dataset:
- `data/RTTBench-Mono-ES.jsonl` — 1,250 synthetic Spanish sentences used for Spanish→Guarani→Spanish RTT.
- `data/gmlgnt.jsonl` — Guarani Global MMLU-Lite examples used by the plain and `lm-eval` evaluation pipelines.

## Models tested

Model configurations are defined in `data/rtt_experiments/es_gn/base_models.json`.

Tested models include:
- Gemma-4: `google/gemma-4-E4B-it`, `google/gemma-4-26B-A4B-it`
- Gemma-3: `google/gemma-3-4b-it`, `google/gemma-3-12b-it`
- Apertus: `swiss-ai/Apertus-8B-Instruct-2509`
- Qwen3: `Qwen/Qwen3-4B-Instruct-2507`
- Meta-Llama-3.1: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- Nemotron-Mini-4B-Instruct: `nvidia/Nemotron-Mini-4B-Instruct`
- Mistral-NeMo-Minitron-8B-Instruct: `nvidia/Mistral-NeMo-Minitron-8B-Instruct`
- Mistral-7B-Instruct-v0.3: `mistralai/Mistral-7B-Instruct-v0.3`
- Grok: `grok-4-fast-non-reasoning` (only for RTT experiments)
- GPT: `gpt-4o-mini` (only for RTT experiments)

## Repository structure

- `src/` — scripts and utility code used for dataset preparation, experiment execution, and evaluation.
- `data/` — input datasets and experiment configuration files.
- `exp/` — experiment configuration files, including RTT, plain Global MMLU-Lite, and `lm-eval` task/model configs.
- `outputs/` — generated model outputs, translations, and evaluation reports.

## Installation

1. Clone the repository.
2. Create and activate a Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

The cosine similarity model used by the evaluator is configured in
`src/eval_rtt_exp.py` as `hackathon-pln-es/paraphrase-spanish-distilroberta`.
Make sure `sentence-transformers` is installed in the active environment before
running RTT evaluation:

```bash
pip install sentence-transformers
```

Install the local harness with Hugging Face support if it is not already
installed in the active environment:

```bash
pip install -e "./lm-evaluation-harness[hf]"
```

Gemma 4 requires a recent `transformers` version. If the environment was created
from an older `requirements.txt`, upgrade Transformers before running Gemma 4:

```bash
pip install -U "transformers>=5.13.1" accelerate
```

### Optional: Language Identification

RTT execution can include language identification metadata for each forward/backward
translation. This is optional. If the module is not installed, RTT still runs and
language-identification are not conducted.

To install only the `language_identifier` module (without cloning the full `corpus`
repository into this project), run:

```bash
rm -rf /tmp/corpus_langid_tmp
git clone --depth 1 --filter=blob:none --sparse https://github.com/guaran-ia/corpus.git /tmp/corpus_langid_tmp
git -C /tmp/corpus_langid_tmp sparse-checkout set src/pipeline/language_identifier
mkdir -p src/corpus/src/pipeline
cp -R /tmp/corpus_langid_tmp/src/pipeline/language_identifier src/corpus/src/pipeline/
rm -rf /tmp/corpus_langid_tmp
```

Verification:

```bash
ls src/corpus/src/pipeline/language_identifier
```

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

---

## RTT Experiments

The RTT workflow is:
1. Translate Spanish source sentences to Guarani.
2. Translate generated Guarani translations back to Spanish.
3. Compare original Spanish sentences with back-translated Spanish sentences.

This design measures how well models preserve meaning and fluency across the Spanish↔Guarani translation round trip.

### Metrics

Evaluation is based on commonly used translation quality metrics and RTT-specific scoring:

- BLEU
- chrF++
- RTTScore (from Zamir et al., [https://arxiv.org/pdf/2601.10804](https://arxiv.org/pdf/2601.10804))

RTTScore is used to enable domain-conditioned evaluation and to better understand how models generalize across domains.

### Running RTT experiments

This section shows the exact commands to run the round-trip translation (RTT) pipeline end-to-end.

#### 1. Configure credentials

`src/run_rtt_exp.py` loads environment variables from `src/.env`.
Rename `src/.env.sample` to `src/.env` and add the keys needed by the models 
you plan to run:

```bash
HF_ACCESS_TOKEN=...
AZURE_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=...
AZURE_OPENAI_DEPLOYMENT=...
```

Notes:
- `HF_ACCESS_TOKEN` is required because the script always calls `huggingface_hub.login(...)`.
- `AZURE_API_KEY` is used for **Grok** (`grok-4-fast-non-reasoning`).
- If you do not run **Grok**, you can keep this key unset and exclude these models in config (see below).
- `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT` are used to invoke 
**Azure OpenAI 5.4**, which is employed for translation validation

#### 2. Set up experiment configuration

Configs live in:
- `data/rtt_experiments/es_gn/config.json` (Spanish→Guarani→Spanish)
- `data/rtt_experiments/en_gn/config.json` (English→Guarani→English)

Each config points to:
- `rtt_data_path`: input dataset, e.g., `data/RTTBench-Mono.jsonl`
- `base_models_list_path`: path to the file containing the list of models to run, e.g., `data/rtt_experiments/es_gn/base_models.json`
- `output_dir`: path to the directory where the experiment outputs should be recorded, e.g., `outputs/rtt_experiment/es_gn`
- `exclude`: list of model variants to be excluded from the running, e.g., `gemma-4-E4B-it`, `gemma-4-26B-A4B-it`

#### 3. Run RTT generation

Spanish↔Guarani:

```bash
python src/run_rtt_exp.py --exp_dir es_gn --batch_size 64
```

English↔Guarani:

```bash
python src/run_rtt_exp.py --exp_dir en_gn --batch_size 64
```

Notes:
- `--exp_dir`: name of the directory inside `data/rtt_experiments` containing the configuration files
- `--batch_size > 0` uses batch inference for Hugging Face models.
- Set `--batch_size 0` to use single-sentence inference mode.
- A timestamped output directory is created automatically, for example:
  `outputs/rtt_experiment/es_gn_YYYYMMDDHHMMSS/`
- Per-model outputs are saved as `*_rtt_results.json`.

#### 4. Run evaluation

After generation finishes, evaluate one result directory:

```bash
python src/eval_rtt_exp.py --res_dir es_gn_YYYYMMDDHHMMSS
```

or

```bash
python src/eval_rtt_exp.py --res_dir en_gn_YYYYMMDDHHMMSS
```

Evaluation updates each `*_rtt_results.json` with metrics and creates:
- `--res_dir`: name of the directory inside `output_dir` (see configuration file) containing the RTT results 
- `overall_evaluation_<res_dir>.csv`

#### 5. Metrics produced

The evaluation script computes:
- sentence-level and corpus-level **SacreBLEU**
- sentence-level and corpus-level **chrF++**
- sentence-level **cosine similarity** with a sentence-transformer embedding model
- RTT-style domain averages (`rtt_sacrebleu`, `rtt_chrf++`)
- translation validity/accounting fields (actual/valid translations and language disagreements)

### Analysis

A notebook with the analyses is available at the `analysis` directory.

### Notes

- The Spanish dataset was generated with **Azure OpenAI GPT-4.1**.
- The experiments aim to reveal how translation quality varies across domains and 
how well models can generalize the Spanish↔Guarani RTT task.
- Outputs and evaluation reports are saved under `outputs/`.

### Blog Article

A medium [blog article](https://jorgesaldivar.medium.com/how-well-do-todays-ai-models-handle-guarani-169b575a48a3) 
was published to present the study and discuss the findigs.

---

## Global MMLU-Lite experiments

Global MMLU-Lite experiments evaluate multiple-choice question answering in
Guarani using `data/gmlgnt.jsonl`.

There are two supported paths:
- the plain project runner in `src/run_mmlu_lite_eval.py`, which asks models to
  generate an answer and parses the generated text;
- the `lm-evaluation-harness` runner in `src/run_lm_eval_exp.py`, which can run
  both likelihood-based and generative `lm-eval` task variants.

### 1. Plain Global MMLU-Lite runner

The plain runner uses the configuration files in `exp/global_mmlu_lite/`:

- `exp/global_mmlu_lite/config.json` — Guarani prompt
- `exp/global_mmlu_lite/config_en.json` — English prompt
- `exp/global_mmlu_lite/config_es.json` — Spanish prompt
- `exp/global_mmlu_lite/base_models.json` — Hugging Face model list

Run the default Guarani prompt evaluation:

```bash
python src/run_mmlu_lite_eval.py \
  --config exp/global_mmlu_lite/config.json \
  --output-dir outputs/global_mmlu_lite \
  --batch-size 16
```

Run a small smoke test:

```bash
python src/run_mmlu_lite_eval.py \
  --config exp/global_mmlu_lite/config.json \
  --output-dir outputs/global_mmlu_lite_smoke \
  --batch-size 1 \
  --max-samples 10
```

Outputs are written to a timestamped directory under the selected output
directory. Per-model prediction files and summary files are created there.

### 2. `lm-eval` setup

The `lm-eval` tasks live in:

- `exp/lm_eval/gn_global_mmlu_lite/gn_global_mmlu_lite.yaml`
- `exp/lm_eval/gn_global_mmlu_lite_generate/gn_global_mmlu_lite_generate.yaml`
- `exp/lm_eval/base_models.json`

Validate that the tasks are discoverable:

```bash
lm_eval validate --include_path exp/lm_eval --tasks gn_global_mmlu_lite
lm_eval validate --include_path exp/lm_eval --tasks gn_global_mmlu_lite_generate
```

### 3. Likelihood-based `lm-eval` task

The task `gn_global_mmlu_lite` uses `output_type: multiple_choice`. It scores the
likelihood of answer labels `A`, `B`, `C`, and `D`; it does not generate text.

For this protocol, do not apply chat templates by default because some
instruction-tuned models show strong answer-label priors after chat formatting:

```bash
python -m src.run_lm_eval_exp \
  --task gn_global_mmlu_lite \
  --run-name gn_global_mmlu_lite_likelihood \
  --no-apply-chat-template \
  --dtype bfloat16 \
  --bootstrap-iters 10000
```

Run one model only:

```bash
python -m src.run_lm_eval_exp \
  --task gn_global_mmlu_lite \
  --run-name diagnostic_gemma4_likelihood \
  --only google/gemma-4-E4B-it \
  --no-apply-chat-template \
  --dtype bfloat16 \
  --limit 50 \
  --bootstrap-iters 0
```

### 4. Generative `lm-eval` task

The task `gn_global_mmlu_lite_generate` uses `output_type: generate_until`. It
asks the model to generate a single answer letter and uses
`exp/lm_eval/gn_global_mmlu_lite_generate/utils.py` to parse outputs such as
`A`, `(B)`, `Answer: C`, or `Mbohovái: D`.

Because this task generates answers rather than scoring label likelihoods, using
the chat template is usually appropriate for instruction-tuned models:

```bash
python -m src.run_lm_eval_exp \
  --task gn_global_mmlu_lite_generate \
  --run-name gn_global_mmlu_lite_generate \
  --apply-chat-template \
  --dtype bfloat16 \
  --bootstrap-iters 10000
```

Run a small diagnostic:

```bash
python -m src.run_lm_eval_exp \
  --task gn_global_mmlu_lite_generate \
  --run-name diagnostic_gemma4_generate \
  --only google/gemma-4-E4B-it \
  --apply-chat-template \
  --dtype bfloat16 \
  --limit 20 \
  --bootstrap-iters 0
```

The `lm-eval` runner writes one directory per model and maintains:
- `run_metadata.json`
- `summary.csv`
- `summary.jsonl`
- per-model `results.json`
- per-model `summary.json`

Use `--skip-existing` with a fixed `--run-name` to resume an interrupted run:

```bash
python -m src.run_lm_eval_exp \
  --task gn_global_mmlu_lite_generate \
  --run-name gn_global_mmlu_lite_generate \
  --skip-existing
```
