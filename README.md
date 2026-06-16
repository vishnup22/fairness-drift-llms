# Release-Level Fairness Drift in Large Language Models

Research code for an anonymous EMNLP 2026 submission on release-level fairness drift in large language models.

This repository implements a release-level fairness regression audit pipeline for measuring how fairness changes across LLM releases. The core question is not simply whether a model is biased, but whether a newer model version is fairer, less fair, or differently biased than the version before it.

## Paper Summary

The study evaluates fairness drift across 12 model versions from five providers and six model families, using seven public bias benchmarks and roughly 200K prompt-response pairs.

The paper defines fairness drift as:

```text
D_f(M_t, M_t+1) = f(M_t+1) - f(M_t)
```

Positive drift means the metric increased. This is desirable for sentiment, but undesirable for toxicity and stereotype scores.

Main findings:

- Toxicity shows a small average decline across newer releases, but individual transitions are non-monotonic.
- Stereotype proxy scores stay flat or increase in most release transitions.
- Gemini 2.5 Pro shows a statistically significant stereotype increase.
- Sentiment, toxicity, and stereotype scores are weakly correlated, with `|r| < 0.2`.
- Excluding StereoSet and CrowS-Pairs preserves the main findings across all 8 release-drift pairs.
- Fairness is multidimensional: safety improvements do not guarantee representational fairness improvements.

## Repository Structure

```text
AI_bias/
|-- src/                           # Pipeline source code
|   |-- config.py                  # Models, API keys, constants
|   |-- data_loader.py             # Dataset loading and preprocessing
|   |-- model_interface.py         # Model API query interfaces
|   |-- fairness_metrics.py        # Fairness metric computation
|   |-- metrics.py                 # Advanced fairness metrics
|   `-- visualization.py           # Plot generation and analysis
|-- data/                          # Static input data
|   `-- crows_pairs_anonymized.csv # Download separately if absent
|-- notebooks/                     # Exploratory analysis
|   `-- qualitative_analysis.ipynb
|-- scripts/                       # Reproduction and paper-figure scripts
|-- tests/                         # Lightweight pipeline tests
|-- docs/                          # Methodology, data, metric, and reproducibility notes
|-- examples/                      # Key figures and result tables from paper
|   |-- plots/                     # Main visualizations
|   `-- results/                   # Summary CSVs and metrics JSON
|-- outputs/                       # Generated at runtime, gitignored
|   |-- results/                   # Raw results and metrics
|   |-- plots/                     # All generated visualizations
|   `-- tables/                    # Summary statistics tables
|-- main.py                        # Benchmark entry point and CLI
|-- requirements.txt
|-- requirements-dev.txt           # Test/development dependencies
|-- pyproject.toml                 # Test configuration and package metadata
|-- .env.example                   # API key template
`-- .gitignore
```

`src/pipeline/` contains the structured execution stages used by `main.py`.
`examples/` is intentionally reserved for representative paper artifacts; runtime outputs go to `outputs/`.

## Models

Release-drift families:

- OpenAI: GPT-4-turbo -> GPT-4o -> GPT-4o-mini -> GPT-4.1 -> GPT-4.1-mini
- Anthropic: Claude Sonnet 4 -> Claude Sonnet 4.5
- Google Gemini: Gemini 2.0 Flash -> 2.5 Flash -> 2.5 Flash Lite -> 2.5 Pro

Contextual scale or cross-tier comparisons:

- Meta LLaMA 3.1: 8B -> 70B -> 405B
- Meta LLaMA 3.2: 1B -> 3B
- Google Gemma 2: 2B -> 9B
- Claude Opus 4 as a parallel cross-tier release

Only release-drift transitions are used for aggregate longitudinal claims.

## Benchmarks

The pipeline evaluates:

- BOLD
- StereoSet
- CrowS-Pairs
- BBQ
- HolisticBias
- WinoBias
- RealToxicityPrompts

CrowS-Pairs must be available at:

```text
data/crows_pairs_anonymized.csv
```

Other datasets are loaded through Hugging Face libraries.

## Metrics

Primary metrics:

- `sentiment_score`: higher is more positive
- `toxicity_score`: lower is better
- `stereotype_score`: lower is better

Additional analysis includes demographic parity, equalized odds, disparate impact ratio, Theil index, KL divergence, Gini coefficient, bootstrap confidence intervals, drift tables, and pairwise statistical tests.

Metric code lives in:

- `src/fairness_metrics.py`
- `src/metrics.py`
- `src/visualization.py`

## Setup

Use Python 3.10 or newer.

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

For development checks:

```bash
pip install -r requirements-dev.txt
```

Create a local `.env` file:

```bash
copy .env.example .env
```

Fill the provider keys you need:

```env
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
HUGGINGFACE_API_KEY=
PERSPECTIVE_API_KEY=
```

`PERSPECTIVE_API_KEY` is optional. Without it, toxicity scoring falls back to a local proxy.

## Running

Sanity check without API calls, using existing scored results:

```bash
python main.py --reuse-results --provider=openai --model=gpt-4-turbo,gpt-4o --dataset=bbq
```

Commercial providers only:

```bash
python main.py --no-hf
```

Open-weight models only:

```bash
python main.py --hf-only
```

Constrained benchmark:

```bash
python main.py --provider=openai,claude --dataset=bold,stereoset
```

Ablation excluding StereoSet and CrowS-Pairs:

```bash
python main.py --reuse-results --exclude-datasets=stereoset,crows_pairs
```

Validate repository structure and imports:

```bash
python scripts/validate_repo.py
pytest
```

## Pipeline

The benchmark runs these stages:

1. Select model families, models, and datasets.
2. Load and normalize benchmark prompts.
3. Run model inference or reuse existing raw/scored results.
4. Filter empty responses, API errors, policy blocks, and refusal-like outputs.
5. Compute or reuse sentiment, toxicity, and stereotype metrics.
6. Compute drift, statistical tests, and aggregate fairness summaries.
7. Generate plots and tables.

Filtering rates are reported because policy blocks and refusals are part of the evaluation signal.

For more detail, see:

- `docs/PIPELINE.md`
- `docs/REPRODUCIBILITY.md`
- `docs/DATASETS.md`
- `docs/METRICS.md`

## Outputs

Runtime artifacts are written to `outputs/`:

- `outputs/results/raw_results_<run_id>.csv`
- `outputs/results/results_with_metrics_<run_id>.csv`
- `outputs/results/comprehensive_metrics_<timestamp>.json`
- `outputs/tables/filtering_rate_by_model.csv`
- `outputs/tables/filtering_rate_by_benchmark.csv`
- `outputs/tables/dataset_model_metric_drift.csv`
- `outputs/tables/drift_by_benchmark.csv`
- `outputs/tables/model_version_summary.csv`
- `outputs/tables/pairwise_statistical_tests.csv`
- `outputs/plots/*.png`

Paper figures and archived result tables live under `examples/`.

## Reproducibility

Important defaults:

- Deterministic decoding: `temperature=0.0`
- Seeded API calls where supported: `seed=42`
- Bootstrap confidence intervals: 1,000 resamples with `seed=42`
- Version ordering is defined in `src/config.py`
- Preserve raw and scored CSVs for any reported paper result

## Limitations

The paper is English-only, uses public benchmark datasets, and relies on automated scoring proxies. Proprietary APIs prevent causal claims about training or post-training changes. Single-turn prompts do not capture multi-turn behavior. Transition-level statistical findings are audit signals for follow-up, not final causal explanations.
