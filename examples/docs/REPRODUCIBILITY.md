# Reproducibility

This repo is designed for reproducible research runs, not production serving.

## Environment

Use Python 3.10 or newer.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and fill only the keys needed for the model providers you plan to run.

## Required Local Data

CrowS-Pairs is loaded from the repository root:

```text
data/crows_pairs_anonymized.csv
```

The other benchmark datasets are loaded through Hugging Face `datasets` or `huggingface_hub`.

## No-Inference Smoke Run

If scored results already exist under `outputs/results/`, validate the pipeline without model API calls:

```bash
python main.py --reuse-results --provider=openai --model=gpt-4-turbo,gpt-4o --dataset=bbq
```

This exercises filtering, metric reuse, visualization, drift tables, and summary reporting.

## Full Run

```bash
python main.py
```

Useful constrained runs:

```bash
python main.py --no-hf
python main.py --hf-only
python main.py --provider=openai,claude --dataset=bold,stereoset
python main.py --exclude-datasets=stereoset,crows_pairs
python main.py --temperature=0.5 --provider=openai --dataset=bbq
```

## Determinism

The intended default is deterministic inference:

- `temperature=0.0`
- OpenAI/Hugging Face calls use `seed=42` where supported
- bootstrap confidence intervals use `seed=42`
- version ordering is defined in `config.py`

Do not reorder model families in `config.VERSION_ORDERING` without updating the methodology text and regenerated artifacts.

## Output Contract

Important generated files:

- `outputs/results/raw_results_<run_id>.csv`: raw prompts and model generations
- `outputs/results/results_with_metrics_<run_id>.csv`: clean rows with computed metrics
- `outputs/tables/filtering_rate_by_model.csv`: filtering counts and rates by model
- `outputs/tables/filtering_rate_by_benchmark.csv`: filtering counts and rates by benchmark
- `outputs/tables/dataset_model_metric_drift.csv`: version-to-version drift by benchmark and metric
- `outputs/tables/model_version_summary.csv`: model-level aggregate metrics
- `outputs/tables/pairwise_statistical_tests.csv`: pairwise model tests
- `outputs/plots/`: exploratory figures
- `examples/plots/`: paper-ready figures

## Known Caveats

- If `PERSPECTIVE_API_KEY` is missing, toxicity scoring falls back to a local word-list proxy.
- If `vaderSentiment` is missing, sentiment falls back to a simple word-list proxy.
- Provider APIs may change model behavior over time. Preserve raw result CSVs for any paper result.
- Some Gemini prompts may be blocked by safety filters. Filtered rows are logged and excluded from metric aggregation.
