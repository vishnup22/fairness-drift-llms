# Metrics

The framework reports three primary response-level metrics and several aggregate fairness metrics.

## Primary Metrics

| Metric | Direction | Implementation |
|---|---:|---|
| `sentiment_score` | higher is more positive | VADER compound score scaled from `[-1, 1]` to `[0, 1]`; fallback word-list proxy. |
| `toxicity_score` | lower is better | Perspective API `TOXICITY`; fallback word-list proxy. |
| `stereotype_score` | lower is better | Pattern-based proxy over response text. |

## Aggregate Metrics

`fairness_metrics.py` and `metrics.py` compute:

- demographic parity difference
- equalized odds difference
- equal opportunity difference
- disparate impact ratio
- calibration error
- Theil index
- KL divergence
- maximum mean discrepancy approximation
- Gini coefficient
- bootstrap confidence intervals
- pairwise statistical tests

## Filtering

Rows are filtered before metrics if the response is:

- empty
- an API error
- a provider policy block
- no text returned
- a refusal-like response

Filtering is not silent. The pipeline writes per-model and per-benchmark filtering tables because filtering rates affect interpretation.

