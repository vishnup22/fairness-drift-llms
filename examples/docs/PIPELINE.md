# Pipeline

The benchmark is organized as explicit stages. `main.py` is intentionally thin and delegates to `src/pipeline/`.

## Stages

1. **Input selection**: choose model families, datasets, and filters.
   - Code: `src/pipeline/execution.py`
   - Configuration: `config.py`

2. **Dataset loading**: load and normalize benchmark examples into records with a `prompt` plus metadata such as `axis`, `group`, `bias_type`, or `target`.
   - Code: `src/data_loader.py`

3. **Inference**: query selected model adapters and save raw generations.
   - Code: `src/model_interface.py`, `src/pipeline/execution.py`
   - Output: `outputs/results/raw_results_<run_id>.csv`

4. **Response filtering**: remove API errors, policy blocks, empty responses, and refusal-like outputs. Filtering rates are reported because they are part of the research signal.
   - Code: `src/pipeline/filtering.py`
   - Outputs: `outputs/tables/filtering_rate_by_model.csv`, `outputs/tables/filtering_rate_by_benchmark.csv`

5. **Scoring**: compute or reuse sentiment, toxicity, and stereotype scores.
   - Code: `src/pipeline/scoring.py`, `src/fairness_metrics.py`, `src/metrics.py`
   - Output: `outputs/results/results_with_metrics_<run_id>.csv`

6. **Artifacts**: generate figures, summary tables, drift tables, and statistical tests.
   - Code: `src/pipeline/reporting.py`, `src/visualization.py`
   - Outputs: `outputs/plots/`, `outputs/tables/`

## Reuse Mode

`--reuse-results` skips inference and loads an existing scored CSV when available. The loader picks the broadest reusable results file by row count, using modification time only as a tiebreaker. This prevents small filtered smoke-test outputs from becoming the default input for later reuse runs.

## Drift Definition

Fairness drift is computed as:

```text
D_f(M_t, M_{t+1}) = f(M_{t+1}) - f(M_t)
```

Positive drift means the metric increased between versions. That is desirable for sentiment, but undesirable for toxicity and stereotype scores.
