# Round-Trip Translation Experiments: The Case of Guarani

This repository contains the scripts, data, and evaluation code for 
benchmarking open-weight base and commercial LLMs on their ability to communicate 
in Guarani.

The experiment uses a round-trip translation (RTT) strategy: Spanish sentences 
are translated to Guarani, then translated back to Spanish. Results are evaluated 
by comparing the original Spanish sentence with the back-translated Spanish sentence.

## Dataset

The experiments are conducted on a synthetic Spanish dataset of 1,250 sentences 
across 25 domains such as Arts, Literature, Business, and Technology.

Dataset:
- `data/RTTBench-Mono-ES.jsonl` — 1,250 synthetic Spanish sentences used for Spanish→Guarani→Spanish RTT.

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

## Installation

1. Clone the repository.
2. Create and activate a Python environment.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Optional: Language Identification

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

## Running RTT experiments

This section shows the exact commands to run the round-trip translation (RTT) pipeline end-to-end.

### 1. Configure credentials

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

### 2. Set up experiment configuration

Configs live in:
- `data/rtt_experiments/es_gn/config.json` (Spanish→Guarani→Spanish)
- `data/rtt_experiments/en_gn/config.json` (English→Guarani→English)

Each config points to:
- `rtt_data_path`: input dataset, e.g., `data/RTTBench-Mono.jsonl`
- `base_models_list_path`: path to the file containing the list of models to run, e.g., `data/rtt_experiments/es_gn/base_models.json`
- `output_dir`: path to the directory where the experiment outputs should be recorded, e.g., `outputs/rtt_experiment/es_gn`
- `exclude`: list of model variants to be excluded from the running, e.g., `gemma-4-E4B-it`, `gemma-4-26B-A4B-it`

### 3. Run RTT generation

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

### 4. Run evaluation

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

### 5. Metrics produced

The evaluation script computes:
- sentence-level and corpus-level **SacreBLEU**
- sentence-level and corpus-level **chrF++**
- RTT-style domain averages (`rtt_sacrebleu`, `rtt_chrf++`)
- translation validity/accounting fields (actual/valid translations and language disagreements)

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

- The Spanish dataset was generated with **Azure OpenAI GPT-4.1**.
- The experiments aim to reveal how translation quality varies across domains and 
how well models can generalize the Spanish↔Guarani RTT task.
- Outputs and evaluation reports are saved under `outputs/`.

## Analysis

A notebook with the analyses is available at the `analysis` directory.

## Article

A medium [blog article](https://jorgesaldivar.medium.com/how-well-do-todays-ai-models-handle-guarani-169b575a48a3) 
was published to present the study and discuss the findigs.
