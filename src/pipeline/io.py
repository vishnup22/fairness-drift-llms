import glob
import os
import re

import pandas as pd

from src.config import HF_MODELS, OUTPUT_DIR


def load_existing_results(include_hf_models=True, hf_only=False):
    """
    Load existing results from OUTPUT_DIR without running model inference.

    Search order:
      1. results_with_metrics_*.csv
      2. combined raw_results_YYYYMMDD_HHMMSS*.csv
      3. concatenated per-model raw_results_*.csv files
    """
    hf_providers = set(HF_MODELS.keys())

    metric_csvs = [
        f for f in glob.glob(f"{OUTPUT_DIR}/results_with_metrics_*.csv")
        if " - Copy" not in f
    ]
    if metric_csvs:
        chosen = _choose_reusable_csv(metric_csvs)
        print(f"  Loading pre-scored results file (metrics already computed): {chosen}")
        df_raw = pd.read_csv(chosen)
    else:
        raw_csvs = [
            f for f in glob.glob(f"{OUTPUT_DIR}/raw_results_*.csv")
            if " - Copy" not in f
        ]
        if not raw_csvs:
            raise RuntimeError(
                f"No results CSVs found in {OUTPUT_DIR}/. "
                "Run without --reuse-results first to generate inference outputs."
            )

        combined = [
            f for f in raw_csvs
            if re.search(r"raw_results_\d{8}_\d{6}", os.path.basename(f))
        ]
        if combined:
            chosen = _choose_reusable_csv(combined)
            print(f"  Loading combined raw results file: {chosen}")
            df_raw = pd.read_csv(chosen)
        else:
            print(f"  Concatenating {len(raw_csvs)} per-model raw result files...")
            df_raw = pd.concat([pd.read_csv(f) for f in sorted(raw_csvs)], ignore_index=True)

    print(
        f"  Loaded {len(df_raw):,} rows | "
        f"{df_raw['model'].nunique()} models | "
        f"{df_raw['provider'].nunique()} providers | "
        f"{df_raw['dataset'].nunique()} datasets"
    )

    if hf_only:
        df_raw = df_raw[df_raw["provider"].isin(hf_providers)].copy()
        print(f"  HF-only filter applied: {len(df_raw):,} rows remain")
    elif not include_hf_models:
        df_raw = df_raw[~df_raw["provider"].isin(hf_providers)].copy()
        print(f"  No-HF filter applied: {len(df_raw):,} rows remain")

    if df_raw.empty:
        raise RuntimeError(
            "No rows remain after provider filter. "
            "Check that outputs/results/ contains data for the requested providers."
        )

    return df_raw


def _choose_reusable_csv(paths):
    """
    Pick the broadest reusable CSV, using mtime only as a tiebreaker.

    Reuse runs can produce filtered subsets. If the next reuse run simply picked
    the newest file, a smoke test could accidentally become the default input.
    """
    return max(paths, key=lambda path: (_csv_data_rows(path), os.path.getmtime(path)))


def _csv_data_rows(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def save_raw_results(df_raw, run_id):
    raw_file = f"{OUTPUT_DIR}/raw_results_{run_id}.csv"
    df_raw.to_csv(raw_file, index=False)
    print(f"Raw results saved: {raw_file}")

    for (provider, model), group_df in df_raw.groupby(["provider", "model"]):
        safe_name = f"{provider}_{model}".replace("/", "_").replace(" ", "_")
        sub_path = f"{OUTPUT_DIR}/raw_results_{safe_name}_{run_id}.csv"
        group_df.to_csv(sub_path, index=False)
        print(f"Saved per-model raw results: {sub_path}")

    return raw_file


def save_metrics_results(df_clean, run_id):
    metrics_file = f"{OUTPUT_DIR}/results_with_metrics_{run_id}.csv"
    df_clean.to_csv(metrics_file, index=False)
    print(f"Results with metrics saved: {metrics_file}")
    return metrics_file
