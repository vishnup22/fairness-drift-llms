import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from src.config import PLOTS_DIR, TABLES_DIR, VERSION_ORDERING

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

def _bootstrap_group_cis(df, group_col, metric, n_bootstrap=1000, seed=42):
    """
    Compute per-group bootstrap 95% CI on *metric*.

    Returns dict: {group_name: (mean, lower_err, upper_err)}
    where lower_err = mean - ci_lower  and  upper_err = ci_upper - mean,
    ready for use as matplotlib yerr/xerr two-row arrays.
    """
    rng = np.random.default_rng(seed)
    result = {}
    for name, group in df.groupby(group_col):
        data = group[metric].dropna().values
        if len(data) == 0:
            result[name] = (np.nan, 0.0, 0.0)
            continue
        boot_means = np.array([
            rng.choice(data, size=len(data), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        mean_val = float(np.mean(data))
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))
        result[name] = (mean_val, mean_val - ci_lower, ci_upper - mean_val)
    return result


def _ordered_models_for_provider(provider_model_set):
    """
    Return models in VERSION_ORDERING sequence.  Models absent from
    VERSION_ORDERING are appended at the end in their original order.
    """
    ordered, seen = [], set()
    for family_models in VERSION_ORDERING.values():
        for m in family_models:
            if m in provider_model_set and m not in seen:
                ordered.append(m)
                seen.add(m)
    for m in provider_model_set:
        if m not in seen:
            ordered.append(m)
    return ordered


# ---------------------------------------------------------------------------
# Fairness drift  D_f(M_t, M_{t+1}) = f(M_{t+1}) - f(M_t)
# ---------------------------------------------------------------------------

def compute_fairness_drift(df, metrics=None):
    """
    Compute version-to-version fairness drift for every family in
    VERSION_ORDERING, using exactly 1,000-sample bootstrap resampling for
    95% confidence intervals.

    Positive drift = metric increased between versions
      (good for sentiment; bad for toxicity and stereotype).

    Writes outputs/tables/dataset_model_metric_drift.csv and returns the DataFrame.
    """
    if metrics is None:
        metrics = ["sentiment_score", "toxicity_score", "stereotype_score"]

    available_models = set(df["model"].unique())
    records = []

    for family, ordered_models in VERSION_ORDERING.items():
        family_models = [m for m in ordered_models if m in available_models]
        if len(family_models) < 2:
            continue

        for i in range(len(family_models) - 1):
            m_t = family_models[i]
            m_t1 = family_models[i + 1]
            df_t = df[df["model"] == m_t]
            df_t1 = df[df["model"] == m_t1]

            for dataset in sorted(df["dataset"].unique()):
                ds_t = df_t[df_t["dataset"] == dataset]
                ds_t1 = df_t1[df_t1["dataset"] == dataset]

                for metric in metrics:
                    if metric not in df.columns:
                        continue
                    vals_t = ds_t[metric].dropna().values
                    vals_t1 = ds_t1[metric].dropna().values
                    if len(vals_t) == 0 or len(vals_t1) == 0:
                        continue

                    rng = np.random.default_rng(42)
                    boot_drifts = np.array([
                        rng.choice(vals_t1, size=len(vals_t1), replace=True).mean()
                        - rng.choice(vals_t, size=len(vals_t), replace=True).mean()
                        for _ in range(1000)
                    ])
                    drift = float(np.mean(vals_t1) - np.mean(vals_t))
                    records.append({
                        "family": family,
                        "model_t": m_t,
                        "model_t1": m_t1,
                        "dataset": dataset,
                        "metric": metric,
                        "drift": round(drift, 6),
                        "ci_lower": round(float(np.percentile(boot_drifts, 2.5)), 6),
                        "ci_upper": round(float(np.percentile(boot_drifts, 97.5)), 6),
                        "n_t": int(len(vals_t)),
                        "n_t1": int(len(vals_t1)),
                        "n_bootstrap": 1000,
                    })

    if not records:
        print("  Drift: no consecutive model pairs found in data — skipping.")
        return pd.DataFrame()

    drift_df = pd.DataFrame(records)
    drift_path = f"{TABLES_DIR}/dataset_model_metric_drift.csv"
    drift_df.to_csv(drift_path, index=False)
    print(f"  Drift table saved: {drift_path}")

    # Per-benchmark drift table — paper-ready column names for Table 4 equivalent
    bench_drift_df = drift_df.rename(columns={
        "dataset": "benchmark",
        "family": "provider",
        "model_t": "model_from",
        "model_t1": "model_to",
    })[["benchmark", "provider", "model_from", "model_to", "metric", "drift", "ci_lower", "ci_upper"]]
    bench_drift_path = f"{TABLES_DIR}/drift_by_benchmark.csv"
    bench_drift_df.to_csv(bench_drift_path, index=False)
    print(f"  Per-benchmark drift table saved: {bench_drift_path}")

    return drift_df

# barplot comparing models
def plot_model_comparison(df, metric="sentiment_score"):
    plt.figure(figsize=(14, 8))

    cis = _bootstrap_group_cis(df, "model", metric)
    model_means = df.groupby("model")[metric].mean().sort_values(ascending=False)
    lower_errs = [cis[m][1] if m in cis else 0 for m in model_means.index]
    upper_errs = [cis[m][2] if m in cis else 0 for m in model_means.index]

    plt.bar(range(len(model_means)), model_means.values,
            yerr=[lower_errs, upper_errs], capsize=5)
    plt.xticks(range(len(model_means)), model_means.index, rotation=45, ha="right")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Model Comparison: {metric.replace('_', ' ').title()} (95% bootstrap CI)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_comparison_{metric}.png", dpi=300)
    plt.close()

# boxplot comparing providers
def plot_provider_comparison(df, metric="sentiment_score"):
    plt.figure(figsize=(12, 8))
    
    sns.boxplot(data=df, x="provider", y=metric, hue="provider", palette="Set2", legend=False)
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(f"Provider Comparison: {metric.replace('_', ' ').title()}")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/provider_comparison_{metric}.png", dpi=300)
    plt.close()

# performance comparison across data
def plot_dataset_comparison(df, metric="sentiment_score"):
    plt.figure(figsize=(14, 8))

    cis = _bootstrap_group_cis(df, "dataset", metric)
    dataset_means = df.groupby("dataset")[metric].mean().sort_values()
    lower_errs = [cis[d][1] if d in cis else 0 for d in dataset_means.index]
    upper_errs = [cis[d][2] if d in cis else 0 for d in dataset_means.index]

    plt.barh(range(len(dataset_means)), dataset_means.values,
             xerr=[lower_errs, upper_errs], capsize=5)
    plt.yticks(range(len(dataset_means)), dataset_means.index)
    plt.xlabel(metric.replace("_", " ").title())
    plt.title(f"Dataset Comparison: {metric.replace('_', ' ').title()} (95% bootstrap CI)")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/dataset_comparison_{metric}.png", dpi=300)
    plt.close()

# heatmap
def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    
    metric_cols = ["sentiment_score", "toxicity_score", "stereotype_score"]
    corr_matrix = df[metric_cols].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0)
    plt.title("Correlation Between Bias Metrics")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/metric_correlation_heatmap.png", dpi=300)
    plt.close()


def plot_axis_level_metrics(df):
    """Plot mean sentiment and toxicity per demographic axis."""
    if "axis" not in df.columns:
        print("Axis column missing; skipping axis-level plot.")
        return

    metric_cols = [col for col in ["sentiment_score", "toxicity_score"] if col in df.columns]
    if not metric_cols:
        return

    axis_means = (
        df.groupby("axis")[metric_cols]
        .mean()
        .sort_values(by=metric_cols[0], ascending=False)
    )

    ax = axis_means.plot(kind="bar", figsize=(14, 7))
    ax.set_title("Mean Sentiment & Toxicity by Axis", fontsize=14, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xlabel("Axis")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/axis_mean_sentiment_toxicity.png", dpi=300)
    plt.close()


def plot_group_level_metrics(df):
    """Plot mean sentiment and toxicity per group within each axis."""
    if not {"axis", "group"}.issubset(df.columns):
        print("Axis/group columns missing; skipping group-level plots.")
        return

    metric_cols = [col for col in ["sentiment_score", "toxicity_score"] if col in df.columns]
    if not metric_cols:
        return

    axes = sorted(df["axis"].dropna().unique())
    for axis in axes:
        axis_df = df[(df["axis"] == axis) & df["group"].notna()]
        if axis_df["group"].nunique() < 2:
            continue
        group_means = (
            axis_df.groupby("group")[metric_cols]
            .mean()
            .sort_values(by=metric_cols[-1], ascending=False)
        )
        ax = group_means.plot(kind="barh", figsize=(14, max(6, 0.4 * len(group_means))))
        ax.set_title(f"{axis}: Group-Level Sentiment & Toxicity", fontsize=14, fontweight="bold")
        ax.set_xlabel("Score")
        ax.set_ylabel("Group")
        plt.tight_layout()
        safe_axis = str(axis).replace("/", "_").replace(" ", "_")
        plt.savefig(f"{PLOTS_DIR}/axis_{safe_axis}_group_sentiment_toxicity.png", dpi=300)
        plt.close()

# Model version comparison within each provider
def plot_provider_model_versions(df, metric="sentiment_score"):
    providers = df["provider"].unique()
    n_providers = len(providers)

    fig, axes = plt.subplots(1, n_providers, figsize=(6 * n_providers, 8))
    if n_providers == 1:
        axes = [axes]

    for idx, provider in enumerate(providers):
        provider_df = df[df["provider"] == provider]
        cis = _bootstrap_group_cis(provider_df, "model", metric)
        model_means = provider_df.groupby("model")[metric].mean().sort_values(ascending=False)
        lower_errs = [cis[m][1] if m in cis else 0 for m in model_means.index]
        upper_errs = [cis[m][2] if m in cis else 0 for m in model_means.index]

        axes[idx].barh(range(len(model_means)), model_means.values,
                       xerr=[lower_errs, upper_errs], capsize=5)
        axes[idx].set_yticks(range(len(model_means)))
        axes[idx].set_yticklabels(model_means.index, fontsize=9)
        axes[idx].set_xlabel(metric.replace("_", " ").title())
        axes[idx].set_title(f"{provider.upper()} Model Versions")
        axes[idx].grid(axis="x", alpha=0.3)

    plt.suptitle(
        f"Model Version Comparison by Provider: {metric.replace('_', ' ').title()} (95% bootstrap CI)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/provider_model_versions_{metric}.png", dpi=300)
    plt.close()

# Heatmap of all models vs all metrics
def plot_model_metrics_heatmap(df):
    metrics = ["sentiment_score", "toxicity_score", "stereotype_score"]
    
    # Create pivot table
    model_means = df.groupby("model")[metrics].mean()
    model_means = model_means.sort_values("sentiment_score", ascending=False)
    
    plt.figure(figsize=(10, max(8, len(model_means) * 0.4)))
    sns.heatmap(model_means.T, annot=True, fmt=".3f", cmap="RdYlGn", 
                center=model_means.mean().mean(), cbar_kws={'label': 'Score'})
    plt.title("All Models vs All Metrics Heatmap", fontsize=14, fontweight='bold')
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Metric", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_metrics_heatmap.png", dpi=300)
    plt.close()

# Detailed model version breakdown
def plot_model_version_radar(df, model_name):
    model_df = df[df["model"] == model_name]
    if len(model_df) == 0:
        return
    
    metrics = ["sentiment_score", "toxicity_score", "stereotype_score"]
    means = [model_df[m].mean() for m in metrics]
    
    # Normalize to 0-1 scale for radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    means += means[:1]  # Complete the circle
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, means, 'o-', linewidth=2, label=model_name)
    ax.fill(angles, means, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim(0, 1)
    ax.set_title(f"Model Performance Profile: {model_name}", fontsize=14, fontweight='bold', pad=20)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_radar_{model_name.replace(' ', '_').replace('/', '_')}.png", dpi=300)
    plt.close()

# Model version detailed comparison table
def create_model_version_table(df, model_version_df=None):
    if model_version_df is None:
        from src.fairness_metrics import compute_model_version_metrics
        model_version_df = compute_model_version_metrics(df)
    
    # Save detailed model version metrics
    model_version_df.to_csv(f"{TABLES_DIR}/model_version_detailed_metrics.csv", index=False)
    
    # Create a summary table with key metrics
    summary_cols = ["provider", "model", "sentiment_mean", "toxicity_mean", 
                    "stereotype_mean", "sentiment_consistency"]
    if all(col in model_version_df.columns for col in summary_cols):
        summary = model_version_df[summary_cols].copy()
        summary = summary.sort_values("sentiment_mean", ascending=False)
        summary.to_csv(f"{TABLES_DIR}/model_version_summary.csv", index=False)
    
    return model_version_df

# Provider comparison with model versions
def plot_provider_model_comparison(df, metric="sentiment_score"):
    plt.figure(figsize=(16, 10))
    
    # Create grouped bar chart - group by provider, show all models within each provider
    providers = sorted(df["provider"].unique())
    provider_models = {}
    
    for provider in providers:
        provider_df = df[df["provider"] == provider]
        models = sorted(provider_df["model"].unique())
        provider_models[provider] = models
    
    # all unique models across providers for consistent coloring
    all_models = sorted(df["model"].unique())
    max_models_per_provider = max(len(models) for models in provider_models.values())
    
    x = np.arange(len(providers))
    width = 0.8 / max_models_per_provider if max_models_per_provider > 0 else 0.2
    
    # colormap for consistent model coloring
    colors = plt.cm.tab20(np.linspace(0, 1, len(all_models)))
    model_color_map = {model: colors[i] for i, model in enumerate(all_models)}
    
    # Plot bars for each provider
    for provider_idx, provider in enumerate(providers):
        provider_df = df[df["provider"] == provider]
        models = sorted(provider_df["model"].unique())
        
        for model_idx, model in enumerate(models):
            model_data = provider_df[provider_df["model"] == model]
            mean_value = model_data[metric].mean()
            
            offset = (model_idx - len(models) / 2 + 0.5) * width
            plt.bar(provider_idx + offset, mean_value, width, 
                   label=model if provider_idx == 0 else "", 
                   color=model_color_map.get(model, 'gray'), alpha=0.8)
    
    plt.xlabel("Provider", fontsize=12)
    plt.ylabel(metric.replace("_", " ").title(), fontsize=12)
    plt.title(f"Provider Comparison with Model Versions: {metric.replace('_', ' ').title()}", 
              fontsize=14, fontweight='bold')
    plt.xticks(x, providers)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/provider_model_comparison_{metric}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Line plot showing model version changes across metrics
def plot_model_version_changes(df, group_by_provider=True):
    metrics = ["sentiment_score", "toxicity_score", "stereotype_score"]
    metric_labels = ["Sentiment Score", "Toxicity Score", "Stereotype Score"]

    def _model_boot_errorbars(model_data, metric):
        """Return (mean, lower_err, upper_err) via bootstrap."""
        data = model_data[metric].dropna().values
        if len(data) == 0:
            return np.nan, 0.0, 0.0
        rng = np.random.default_rng(42)
        boot_means = np.array([
            rng.choice(data, size=len(data), replace=True).mean()
            for _ in range(1000)
        ])
        mean_val = float(np.mean(data))
        return mean_val, mean_val - float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5)) - mean_val

    if group_by_provider:
        providers = sorted(df["provider"].unique())

        for provider in providers:
            provider_df = df[df["provider"] == provider]
            if len(provider_df) == 0:
                continue

            # Use VERSION_ORDERING so models appear in correct longitudinal sequence
            models = _ordered_models_for_provider(set(provider_df["model"].unique()))
            if len(models) < 2:
                continue

            plt.figure(figsize=(14, 8))
            x_positions = np.arange(len(metrics))
            colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

            for idx, model in enumerate(models):
                model_data = provider_df[provider_df["model"] == model]
                means, lower_errs, upper_errs = [], [], []
                for metric in metrics:
                    m, lo, hi = _model_boot_errorbars(model_data, metric)
                    means.append(m)
                    lower_errs.append(lo)
                    upper_errs.append(hi)

                plt.plot(x_positions, means, marker="o", linewidth=2.5,
                         markersize=8, label=model, color=colors[idx], alpha=0.8)
                plt.errorbar(x_positions, means,
                             yerr=[lower_errs, upper_errs],
                             fmt="none", color=colors[idx], alpha=0.5, capsize=5)

            plt.xticks(x_positions, metric_labels, fontsize=11)
            plt.ylabel("Score", fontsize=12)
            plt.title(f"Model Version Changes: {provider.upper()} (95% bootstrap CI)",
                      fontsize=14, fontweight="bold")
            plt.legend(loc="best", fontsize=9, framealpha=0.9)
            plt.grid(True, alpha=0.3, linestyle="--")
            plt.ylim(bottom=0)
            plt.tight_layout()
            plt.savefig(f"{PLOTS_DIR}/model_version_changes_{provider}.png", dpi=300)
            plt.close()

    # Overall comparison — all providers
    plt.figure(figsize=(16, 10))
    all_models = _ordered_models_for_provider(set(df["model"].unique()))
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(all_models), 1)))
    model_color_map = {model: colors[i] for i, model in enumerate(all_models)}
    x_positions = np.arange(len(metrics))

    for model in all_models:
        model_data = df[df["model"] == model]
        if len(model_data) == 0:
            continue
        means, lower_errs, upper_errs = [], [], []
        for metric in metrics:
            m, lo, hi = _model_boot_errorbars(model_data, metric)
            means.append(m)
            lower_errs.append(lo)
            upper_errs.append(hi)

        plt.plot(x_positions, means, marker="o", linewidth=2, markersize=7,
                 label=model, color=model_color_map[model], alpha=0.7)
        plt.errorbar(x_positions, means, yerr=[lower_errs, upper_errs],
                     fmt="none", color=model_color_map[model], alpha=0.4, capsize=4)

    plt.xticks(x_positions, metric_labels, fontsize=11)
    plt.ylabel("Score", fontsize=12)
    plt.title("Model Version Changes Across All Providers (95% bootstrap CI)",
              fontsize=16, fontweight="bold")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8, ncol=1)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_version_changes_all.png", dpi=300, bbox_inches="tight")
    plt.close()

# Line plot showing version progression (if version numbers can be extracted)
def plot_model_version_progression(df, metric="sentiment_score"):
    
    # Extract base model names and try to order by version
    def extract_base_name(model_name):
        """Extract base model name (e.g., 'gpt-4' from 'gpt-4.1-mini')"""
        parts = model_name.split('-')
        if len(parts) >= 2:
            return '-'.join(parts[:2])
        return parts[0] if parts else model_name
    
    def extract_version_number(model_name):
        """Try to extract version number for ordering"""
        import re
        # Look for version patterns like 4.1, 3.1, 2.5, etc.
        match = re.search(r'(\d+\.\d+)', model_name)
        if match:
            return float(match.group(1))
        # Look for single digit versions
        match = re.search(r'[vV]?(\d+)', model_name)
        if match:
            return float(match.group(1))
        return 0.0
    
    # Group models by base name
    df_copy = df.copy()
    df_copy['base_name'] = df_copy['model'].apply(extract_base_name)
    df_copy['version_num'] = df_copy['model'].apply(extract_version_number)
    
    base_names = df_copy['base_name'].unique()
    
    # Filter to only base names with at least 2 models
    valid_base_names = []
    for base_name in sorted(base_names):
        base_df = df_copy[df_copy['base_name'] == base_name]
        models = base_df['model'].unique()
        if len(models) >= 2:
            valid_base_names.append(base_name)
    
    if len(valid_base_names) == 0:
        return  # No valid base names with multiple versions
    
    fig, axes = plt.subplots(1, len(valid_base_names), figsize=(6*len(valid_base_names), 8))
    if len(valid_base_names) == 1:
        axes = [axes]
    
    for idx, base_name in enumerate(valid_base_names):
        base_df = df_copy[df_copy['base_name'] == base_name]
        models = base_df['model'].unique()
        
        # Sort models by version number
        model_version_pairs = [(m, extract_version_number(m)) for m in models]
        model_version_pairs.sort(key=lambda x: x[1])
        sorted_models = [m for m, v in model_version_pairs]
        
        # Bootstrap 95% CI via the shared helper (n=1000, seed=42)
        cis = _bootstrap_group_cis(base_df, "model", metric)
        model_means = [cis[m][0] if m in cis else np.nan for m in sorted_models]
        ci_lower_abs = [cis[m][0] - cis[m][1] if m in cis else np.nan for m in sorted_models]
        ci_upper_abs = [cis[m][0] + cis[m][2] if m in cis else np.nan for m in sorted_models]
        lower_errs = [cis[m][1] if m in cis else 0.0 for m in sorted_models]
        upper_errs = [cis[m][2] if m in cis else 0.0 for m in sorted_models]

        # Plot line + shaded CI band + cap markers
        x_positions = np.arange(len(sorted_models))
        axes[idx].plot(x_positions, model_means, marker="o", linewidth=2.5,
                       markersize=10, color="steelblue", alpha=0.8)
        axes[idx].fill_between(x_positions, ci_lower_abs, ci_upper_abs,
                               color="steelblue", alpha=0.15, label="95% CI")
        axes[idx].errorbar(x_positions, model_means,
                           yerr=[lower_errs, upper_errs],
                           fmt="none", color="steelblue", alpha=0.5, capsize=5)
        axes[idx].set_xticks(x_positions)
        axes[idx].set_xticklabels(sorted_models, rotation=45, ha='right', fontsize=9)
        axes[idx].set_ylabel(metric.replace("_", " ").title(), fontsize=11)
        axes[idx].set_title(f"{base_name.upper()} Version Progression", 
                           fontsize=12, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, linestyle='--')
        axes[idx].set_ylim(bottom=0)
    
    plt.suptitle(f"Model Version Progression: {metric.replace('_', ' ').title()}", 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/model_version_progression_{metric}.png", dpi=300)
    plt.close()

# summary 
def create_summary_table(df):
    summary = df.groupby(["provider", "model"]).agg({
        "sentiment_score": ["mean", "std", "min", "max"],
        "toxicity_score": ["mean", "std", "min", "max"],
        "stereotype_score": ["mean", "std", "min", "max"]
    }).round(4)
    
    summary.columns = ["_".join(col).strip() for col in summary.columns.values]
    summary.to_csv(f"{TABLES_DIR}/summary_statistics.csv")
    return summary

# ANOVA & Post-hoc tests
def perform_statistical_tests(df):
    results = {}
    
    # ANOVA for provider differences
    providers = df["provider"].unique()
    groups = [df[df["provider"] == p]["sentiment_score"].dropna().values for p in providers]
    groups = [group for group in groups if len(group) > 1]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        results["provider_anova"] = {
            "f_statistic": f_stat,
            "p_value": p_value,
            "significant": p_value < 0.05
        }
    else:
        results["provider_anova"] = {
            "skipped": True,
            "reason": "requires at least two providers with at least two scored rows each",
        }
    
    # Pairwise t-tests across all models
    all_models = df["model"].unique()
    pairwise_results = []

    for i, model1 in enumerate(all_models):
        for model2 in all_models[i+1:]:
            group1 = df[df["model"] == model1]["sentiment_score"]
            group2 = df[df["model"] == model2]["sentiment_score"]
            t_stat, p_val = stats.ttest_ind(group1, group2)
            pairwise_results.append({
                "model1": model1,
                "model2": model2,
                "t_statistic": t_stat,
                "p_value": p_val,
                "significant": p_val < 0.05
            })
    
    results["pairwise_tests"] = pd.DataFrame(pairwise_results)
    results["pairwise_tests"].to_csv(f"{TABLES_DIR}/pairwise_statistical_tests.csv", index=False)

    # Axis-level ANOVA on toxicity
    if "axis" in df.columns and "toxicity_score" in df.columns:
        axis_values = df["axis"].dropna().unique()
        if len(axis_values) >= 2:
            axis_groups = [df[df["axis"] == axis]["toxicity_score"].dropna().values for axis in axis_values]
            if all(len(group) > 1 for group in axis_groups):
                f_axis, p_axis = stats.f_oneway(*axis_groups)
                results["axis_toxicity_anova"] = {
                    "f_statistic": f_axis,
                    "p_value": p_axis,
                    "significant": p_axis < 0.05,
                    "axes_tested": axis_values.tolist(),
                }

    # Group-level ANOVAs within each axis
    if {"axis", "group", "toxicity_score"}.issubset(df.columns):
        group_results = {}
        for axis in df["axis"].dropna().unique():
            axis_subset = df[(df["axis"] == axis) & df["group"].notna()]
            if axis_subset["group"].nunique() < 2:
                continue
            group_arrays = [
                axis_subset[axis_subset["group"] == group]["toxicity_score"].dropna().values
                for group in axis_subset["group"].unique()
            ]
            if any(len(arr) < 2 for arr in group_arrays):
                continue
            f_val, p_val = stats.f_oneway(*group_arrays)
            group_results[axis] = {
                "f_statistic": f_val,
                "p_value": p_val,
                "significant": p_val < 0.05,
                "groups_tested": axis_subset["group"].unique().tolist(),
            }
        if group_results:
            results["axis_group_toxicity_anova"] = group_results
    
    return results


def create_advanced_fairness_table(df):
    """
    Produce outputs/tables/advanced_fairness_metrics.csv — one row per (model, benchmark).

    Per-row columns:
      model, benchmark, provider, n_responses
      sentiment_score_mean/std, toxicity_score_mean/std, stereotype_score_mean/std

    Per-benchmark aggregate columns (same value repeated for every model in that
    benchmark — allows readers to compare model-level means against the
    benchmark-wide fairness spread):
      demographic_parity_diff_{metric}   — max positive-rate gap across models
      disparate_impact_ratio_{metric}    — min/max positive-rate ratio across models
      theil_index_{metric}               — Theil inequality index across models
      gini_coefficient_{metric}          — Gini coefficient of per-model means
      mean_kl_divergence_{metric}        — mean pairwise KL divergence across models
    """
    from src.metrics import compute_demographic_parity_difference, compute_disparate_impact_ratio
    from src.fairness_metrics import compute_theil_index, compute_kl_divergence_between_groups, compute_gini_coefficient
    import warnings

    outcome_cols = [c for c in ["sentiment_score", "toxicity_score", "stereotype_score"] if c in df.columns]
    records = []

    for dataset in sorted(df["dataset"].unique()):
        ds_df = df[df["dataset"] == dataset]

        # Compute benchmark-level across-model fairness metrics once per dataset
        bench_agg = {}
        if ds_df["model"].nunique() >= 2:
            for col in outcome_cols:
                dpd = compute_demographic_parity_difference(ds_df, "model", outcome_col=col)
                bench_agg[f"demographic_parity_diff_{col}"] = round(float(dpd), 6) if dpd is not None else None

                dir_r = compute_disparate_impact_ratio(ds_df, "model", outcome_col=col)
                bench_agg[f"disparate_impact_ratio_{col}"] = round(float(dir_r), 6) if dir_r is not None else None

                theil = compute_theil_index(ds_df, "model", col)
                bench_agg[f"theil_index_{col}"] = round(float(theil), 6) if theil is not None else None

                kl = compute_kl_divergence_between_groups(ds_df, "model", col)
                bench_agg[f"mean_kl_divergence_{col}"] = (
                    round(float(kl["mean_kl_divergence"]), 6) if kl is not None else None
                )

                model_means_arr = ds_df.groupby("model")[col].mean()
                if len(model_means_arr) > 1:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        gini = compute_gini_coefficient(model_means_arr.values)
                    bench_agg[f"gini_coefficient_{col}"] = round(float(gini), 6) if gini is not None else None

        for model in sorted(ds_df["model"].unique()):
            model_ds = ds_df[ds_df["model"] == model]
            row = {
                "model": model,
                "benchmark": dataset,
                "provider": model_ds["provider"].iloc[0],
                "n_responses": len(model_ds),
            }
            for col in outcome_cols:
                row[f"{col}_mean"] = round(float(model_ds[col].mean()), 6)
                row[f"{col}_std"] = round(float(model_ds[col].std()), 6)
            row.update(bench_agg)
            records.append(row)

    if not records:
        print("  Advanced fairness table: no data — skipping.")
        return pd.DataFrame()

    adv_df = pd.DataFrame(records)
    path = f"{TABLES_DIR}/advanced_fairness_metrics.csv"
    adv_df.to_csv(path, index=False)
    print(f"  Advanced fairness metrics table saved: {path}")
    return adv_df


def generate_all_visualizations(df, model_version_df=None):
    print("\nGenerating visualizations")

    # Fairness drift — must run before plots so CSV is ready
    print("Computing fairness drift (bootstrap, n=1000)...")
    compute_fairness_drift(df)
    
    metrics = ["sentiment_score", "toxicity_score", "stereotype_score"]
    
    # Basic comparisons
    for metric in metrics:
        plot_model_comparison(df, metric)
        plot_provider_comparison(df, metric)
        plot_dataset_comparison(df, metric)
        plot_provider_model_versions(df, metric)
        plot_provider_model_comparison(df, metric)
    
    # Advanced visualizations
    plot_correlation_heatmap(df)
    plot_model_metrics_heatmap(df)
    plot_axis_level_metrics(df)
    plot_group_level_metrics(df)
    
    # Model version line plots
    print("\nGenerating model version line plots...")
    try:
        plot_model_version_changes(df, group_by_provider=True)
        for metric in metrics:
            plot_model_version_progression(df, metric)
    except Exception as e:
        print(f"  Warning: Could not generate some model version plots: {e}")
    
    # Create detailed tables
    summary = create_summary_table(df)
    model_version_table = create_model_version_table(df, model_version_df)
    create_advanced_fairness_table(df)
    
    # Generate radar charts for top models
    top_models = df.groupby("model")["sentiment_score"].mean().nlargest(10).index
    print(f"\nGenerating radar charts for top {len(top_models)} models...")
    for model in top_models:
        try:
            plot_model_version_radar(df, model)
        except Exception as e:
            print(f"  Warning: Could not generate radar chart for {model}: {e}")
    
    # Statistical tests
    stats_results = perform_statistical_tests(df)
    
    print(f"\nVisualizations saved to {PLOTS_DIR}/")
    print(f"Tables saved to {TABLES_DIR}/")
    print(f"   - summary_statistics.csv")
    print(f"   - model_version_detailed_metrics.csv")
    print(f"   - model_version_summary.csv")
    print(f"   - pairwise_statistical_tests.csv")
    print(f"   - dataset_model_metric_drift.csv")
    print(f"   - drift_by_benchmark.csv")
    print(f"   - advanced_fairness_metrics.csv")
    
    return summary, stats_results
