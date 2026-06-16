from datetime import datetime
import json

from src.config import OUTPUT_DIR
from src.fairness_metrics import compute_all_fairness_metrics

try:
    from src.metrics import compute_comprehensive_fairness_metrics
    COMPREHENSIVE_METRICS_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_METRICS_AVAILABLE = False


METRIC_COLS = ["sentiment_score", "toxicity_score", "stereotype_score"]


def score_clean_results(df_clean, total_filtered, filter_rate):
    if all(col in df_clean.columns for col in METRIC_COLS):
        print("\nMetric columns detected in loaded data - skipping API recomputation.")
        from src.fairness_metrics import compute_model_version_metrics

        model_version_df = compute_model_version_metrics(df_clean)
        metrics = {
            "mean_sentiment": float(df_clean["sentiment_score"].mean()),
            "std_sentiment": float(df_clean["sentiment_score"].std()),
            "mean_toxicity": float(df_clean["toxicity_score"].mean()),
            "std_toxicity": float(df_clean["toxicity_score"].std()),
            "mean_stereotype": float(df_clean["stereotype_score"].mean()),
            "std_stereotype": float(df_clean["stereotype_score"].std()),
            "sentiment_method": "precomputed",
            "toxicity_method": "precomputed",
            "model_version_details": model_version_df,
        }
        for provider in df_clean["provider"].unique():
            provider_df = df_clean[df_clean["provider"] == provider]
            metrics[f"{provider}_mean_sentiment"] = float(provider_df["sentiment_score"].mean())
            metrics[f"{provider}_mean_toxicity"] = float(provider_df["toxicity_score"].mean())
            metrics[f"{provider}_mean_stereotype"] = float(provider_df["stereotype_score"].mean())
    else:
        print("\nComputing fairness and bias metrics:")
        df_clean, metrics = compute_all_fairness_metrics(df_clean)

    metrics["filtered_rows"] = total_filtered
    metrics["filter_rate"] = filter_rate

    if COMPREHENSIVE_METRICS_AVAILABLE:
        _attach_comprehensive_metrics(df_clean, metrics)
    else:
        print("Comprehensive metrics not available. Using basic metrics.")

    return df_clean, metrics


def _attach_comprehensive_metrics(df_clean, metrics):
    print("Computing comprehensive fairness metrics:")
    comprehensive_metrics = {
        "provider_fairness": compute_comprehensive_fairness_metrics(
            df_clean, group_col="provider", outcome_col="sentiment_score"
        ),
        "model_fairness": compute_comprehensive_fairness_metrics(
            df_clean, group_col="model", outcome_col="sentiment_score"
        ),
    }

    comprehensive_metrics_file = f"{OUTPUT_DIR}/comprehensive_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    serializable = {}
    for key, value in comprehensive_metrics.items():
        if isinstance(value, dict):
            serializable[key] = {}
            for sub_key, sub_value in value.items():
                if hasattr(sub_value, "to_dict"):
                    serializable[key][sub_key] = sub_value.to_dict("records")
                else:
                    serializable[key][sub_key] = sub_value

    with open(comprehensive_metrics_file, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    print(f"Comprehensive metrics saved: {comprehensive_metrics_file}")
    metrics["comprehensive_metrics"] = comprehensive_metrics
