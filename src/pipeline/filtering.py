import pandas as pd

from src.config import TABLES_DIR


REFUSAL_PATTERNS = [
    "i cannot",
    "i'm sorry, i'm unable",
    "i'm not able to",
    "i am unable to",
    "as an ai, i cannot",
    "as an ai i cannot",
    "i apologize, but i",
    "i'm afraid i can't",
]


def classify_response(response):
    """
    Classify a model response for filtering.

    Returns (should_filter: bool, reason: str | None).
    reason is None when the response is clean.
    """
    if not isinstance(response, str) or response == "":
        return True, "empty"
    if response.startswith("Error:"):
        return True, "api_error"
    if response.startswith("[BLOCKED:"):
        return True, "policy_blocked"
    if response == "[No text returned]":
        return True, "no_text"

    response_lower = response.lower()
    for pattern in REFUSAL_PATTERNS:
        if pattern in response_lower:
            return True, "refusal"

    return False, None


def apply_response_filtering(df_raw):
    """
    Add filtering metadata, write filtering-rate tables, and return clean rows.

    Returns:
        df_clean, total_filtered, filter_rate
    """
    df = df_raw.copy()
    df["filter_reason"] = [classify_response(response)[1] for response in df["response"]]
    df["is_filtered"] = df["filter_reason"].notna()

    filtered_df = df[df["is_filtered"]].copy()
    total_filtered = int(df["is_filtered"].sum())
    total_rows = len(df)
    filter_rate = float(total_filtered / total_rows) if total_rows else 0.0

    if total_filtered:
        print(f"\nWARN: Filtered {total_filtered} responses ({filter_rate:.2%}):")
        for _, row in filtered_df.iterrows():
            print(
                f"    [{row['filter_reason']}] provider={row['provider']} "
                f"model={row['model']} dataset={row['dataset']} "
                f"prompt_id={row['prompt_id']} | "
                f"response_prefix={str(row['response'])[:60]!r}"
            )

    _save_filtering_rate_by_model(df)
    _save_filtering_rate_by_benchmark(df)

    df_clean = df[~df["is_filtered"]].drop(columns=["is_filtered", "filter_reason"]).copy()
    if df_clean.empty:
        raise RuntimeError("All generations were filtered. Cannot compute metrics.")

    return df_clean, total_filtered, filter_rate


def _save_filtering_rate_by_model(df):
    records = []

    filtered = df[df["is_filtered"]]
    for (provider, model, reason), group in filtered.groupby(["provider", "model", "filter_reason"]):
        records.append({
            "provider": provider,
            "model": model,
            "filter_reason": reason,
            "count": len(group),
        })

    for (provider, model), group in df.groupby(["provider", "model"]):
        total = len(group)
        n_filtered = int(group["is_filtered"].sum())
        records.extend([
            {
                "provider": provider,
                "model": model,
                "filter_reason": "_total_filtered",
                "count": n_filtered,
            },
            {
                "provider": provider,
                "model": model,
                "filter_reason": "_total_responses",
                "count": total,
            },
            {
                "provider": provider,
                "model": model,
                "filter_reason": "_filter_rate_pct",
                "count": round(100 * n_filtered / total, 2) if total else 0,
            },
        ])

    if records:
        path = f"{TABLES_DIR}/filtering_rate_by_model.csv"
        pd.DataFrame(records).sort_values(["provider", "model", "filter_reason"]).to_csv(path, index=False)
        print(f"Filtering rate table saved: {path}")


def _save_filtering_rate_by_benchmark(df):
    records = []

    filtered = df[df["is_filtered"]]
    for (dataset, reason), group in filtered.groupby(["dataset", "filter_reason"]):
        records.append({
            "benchmark": dataset,
            "filter_reason": reason,
            "count": len(group),
        })

    for dataset, group in df.groupby("dataset"):
        total = len(group)
        n_filtered = int(group["is_filtered"].sum())
        records.extend([
            {"benchmark": dataset, "filter_reason": "_total_filtered", "count": n_filtered},
            {"benchmark": dataset, "filter_reason": "_total_responses", "count": total},
            {
                "benchmark": dataset,
                "filter_reason": "_filter_rate_pct",
                "count": round(100 * n_filtered / total, 2) if total else 0,
            },
        ])

    if records:
        path = f"{TABLES_DIR}/filtering_rate_by_benchmark.csv"
        pd.DataFrame(records).sort_values(["benchmark", "filter_reason"]).to_csv(path, index=False)
        print(f"Filtering rate by benchmark saved: {path}")
