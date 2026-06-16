"""
Generate all 8 publication-ready figures for the EMNLP paper submission.
Data source: outputs/results/results_with_metrics_20251115_035107.csv
All figures saved to examples/plots/ as both PDF (vector, fonts embedded) and PNG 300 DPI.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from scipy import stats

# ── publication style ─────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")

mpl.rcParams.update({
    # Font embedding — required for IEEE/ACL/ACM submission
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    # Sizes matching user spec
    "axes.titlesize":    12,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "legend.title_fontsize": 9,
    # Spines
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_FILE   = "outputs/results/results_with_metrics_20251115_035107.csv"
ABLATION_TABLE = "outputs/tables/ablation_comparison.csv"
OUTPUT_DIR     = "examples/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── taxonomy ──────────────────────────────────────────────────────────────────
VERSION_ORDERING = {
    "openai":        ["gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini"],
    "claude_sonnet": ["claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929"],
    "claude_opus":   ["claude-opus-4-1-20250805"],
    "gemini":        ["gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    "llama31":       ["meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-70B-Instruct",
                      "meta-llama/Llama-3.1-405B-Instruct"],
    "llama32":       ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"],
    "gemma":         ["google/gemma-2-2b-it", "google/gemma-2-9b-it"],
}

FAMILY_DISPLAY = {
    "openai":        "OpenAI",
    "claude_sonnet": "Claude (Sonnet)",
    "claude_opus":   "Claude (Opus)",
    "gemini":        "Gemini",
    "llama31":       "LLaMA-3.1",
    "llama32":       "LLaMA-3.2",
    "gemma":         "Gemma-2",
}

FAMILY_COLORS = {
    "openai":        "#1f77b4",
    "claude_sonnet": "#ff7f0e",
    "claude_opus":   "#d62728",
    "gemini":        "#2ca02c",
    "llama31":       "#9467bd",
    "llama32":       "#8c564b",
    "gemma":         "#e377c2",
}

PROVIDER_COLORS = {
    "claude":  "#ff7f0e",
    "openai":  "#1f77b4",
    "gemini":  "#2ca02c",
    "llama31": "#9467bd",
    "llama32": "#8c564b",
    "gemma":   "#e377c2",
}

SHORT_LABEL = {
    "gpt-4-turbo":                         "GPT-4\nturbo",
    "gpt-4o":                              "GPT-4o",
    "gpt-4o-mini":                         "GPT-4o\nmini",
    "gpt-4.1":                             "GPT-4.1",
    "gpt-4.1-mini":                        "GPT-4.1\nmini",
    "claude-sonnet-4-20250514":            "Sonnet-4\n(May)",
    "claude-sonnet-4-5-20250929":          "Sonnet-4.5\n(Sep)",
    "claude-opus-4-1-20250805":            "Opus-4.1\n(Aug)",
    "gemini-2.0-flash":                    "2.0\nFlash",
    "gemini-2.5-flash":                    "2.5\nFlash",
    "gemini-2.5-flash-lite":               "2.5 Flash\nLite",
    "gemini-2.5-pro":                      "2.5\nPro",
    "meta-llama/Llama-3.1-8B-Instruct":   "8B",
    "meta-llama/Llama-3.1-70B-Instruct":  "70B",
    "meta-llama/Llama-3.1-405B-Instruct": "405B",
    "meta-llama/Llama-3.2-1B-Instruct":   "1B",
    "meta-llama/Llama-3.2-3B-Instruct":   "3B",
    "google/gemma-2-2b-it":               "2B",
    "google/gemma-2-9b-it":               "9B",
}

METRIC_COLS    = ["sentiment_score", "toxicity_score", "stereotype_score"]
METRIC_DISPLAY = {
    "sentiment_score":  "Sentiment",
    "toxicity_score":   "Toxicity",
    "stereotype_score": "Stereotype",
}
METRIC_COLORS = {
    "sentiment_score":  "#2c7bb6",
    "toxicity_score":   "#d7191c",
    "stereotype_score": "#fdae61",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _bootstrap_ci(values, n=1000, seed=42, alpha=0.05):
    """Return (mean, ci_lo, ci_hi) via n=1000-sample bootstrap, seed=42."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng  = np.random.default_rng(seed)
    boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                     for _ in range(n)])
    return (float(vals.mean()),
            float(np.percentile(boot, 100 * alpha / 2)),
            float(np.percentile(boot, 100 * (1 - alpha / 2))))


def _save_fig(fig, stem):
    """Save fig as both PDF (fonts embedded) and PNG 300 DPI to OUTPUT_DIR."""
    pdf_path = f"{OUTPUT_DIR}/{stem}.pdf"
    png_path = f"{OUTPUT_DIR}/{stem}.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight", format="pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", format="png")
    plt.close(fig)
    sizes = (os.path.getsize(pdf_path), os.path.getsize(png_path))
    print(f"  Saved {stem}.pdf ({sizes[0]:,}B)  +  .png ({sizes[1]:,}B)")


def _compute_drift_table(df):
    """
    Compute signed version-to-version drift for all consecutive pairs in
    VERSION_ORDERING.  Matches visualization.compute_fairness_drift logic
    (n=1000 bootstrap, seed=42).
    """
    available = set(df["model"].unique())
    records   = []
    for family, ordered in VERSION_ORDERING.items():
        fam_models = [m for m in ordered if m in available]
        if len(fam_models) < 2:
            continue
        for i in range(len(fam_models) - 1):
            m_t, m_t1 = fam_models[i], fam_models[i + 1]
            df_t, df_t1 = df[df["model"] == m_t], df[df["model"] == m_t1]
            for dataset in sorted(df["dataset"].unique()):
                for metric in METRIC_COLS:
                    vals_t  = df_t[df_t["dataset"] == dataset][metric].dropna().values
                    vals_t1 = df_t1[df_t1["dataset"] == dataset][metric].dropna().values
                    if len(vals_t) == 0 or len(vals_t1) == 0:
                        continue
                    records.append({
                        "family":  family,
                        "metric":  metric,
                        "drift":   float(vals_t1.mean() - vals_t.mean()),
                    })
    return pd.DataFrame(records)


# ── FIG 1: Aggregate Drift Bar Chart ─────────────────────────────────────────

def make_fig1(df):
    """
    Three bars — one per metric — showing mean |absolute drift| across all
    version transitions and datasets, with 95% bootstrap CI error bars.
    figsize=(7, 4).
    """
    drift_df = _compute_drift_table(df)
    means, lo_errs, hi_errs = [], [], []
    for metric in METRIC_COLS:
        abs_d = drift_df[drift_df["metric"] == metric]["drift"].abs().dropna().values
        mn, lo, hi = _bootstrap_ci(abs_d)
        means.append(mn)
        lo_errs.append(mn - lo)
        hi_errs.append(hi - mn)

    fig, ax = plt.subplots(figsize=(7, 4))
    x      = np.arange(len(METRIC_COLS))
    colors = [METRIC_COLORS[m] for m in METRIC_COLS]
    labels = [METRIC_DISPLAY[m] for m in METRIC_COLS]

    ax.bar(x, means, yerr=[lo_errs, hi_errs], capsize=8, width=0.5,
           color=colors, alpha=0.85, edgecolor="black", linewidth=0.8,
           error_kw={"elinewidth": 1.5, "ecolor": "black"})

    for xi, (mn, hi_e) in enumerate(zip(means, hi_errs)):
        ax.text(xi, mn + hi_e + max(means) * 0.01, f"{mn:.4f}",
                ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean |Drift| across all version transitions")
    ax.set_title("Aggregate Fairness Drift by Metric with 95% Bootstrap CI",
                 fontweight="bold")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save_fig(fig, "fig1_aggregate_drift")


# ── FIG 2: Provider-Level Fairness Profile ───────────────────────────────────

def make_fig2(df):
    """
    Grouped bar chart: provider × metric, all three metrics per provider,
    with 95% bootstrap CI error bars.  figsize=(7, 4).
    """
    PROV_DISPLAY = {
        "claude": "Claude", "openai": "OpenAI", "gemini": "Gemini",
        "llama31": "LLaMA-3.1", "llama32": "LLaMA-3.2", "gemma": "Gemma-2",
    }
    providers = sorted(df["provider"].unique(), key=lambda p: PROV_DISPLAY.get(p, p))
    n_prov    = len(providers)
    n_met     = len(METRIC_COLS)
    bar_w     = 0.72 / n_met

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(n_prov)

    for mi, metric in enumerate(METRIC_COLS):
        offset = (mi - (n_met - 1) / 2) * bar_w
        means, lo_errs, hi_errs = [], [], []
        for prov in providers:
            vals = df[df["provider"] == prov][metric].dropna().values
            mn, lo, hi = _bootstrap_ci(vals)
            means.append(mn)
            lo_errs.append(mn - lo)
            hi_errs.append(hi - mn)

        ax.bar(x + offset, means, bar_w,
               yerr=[lo_errs, hi_errs], capsize=3,
               label=METRIC_DISPLAY[metric], color=METRIC_COLORS[metric],
               alpha=0.82, edgecolor="white", linewidth=0.4,
               error_kw={"elinewidth": 1.2, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels([PROV_DISPLAY.get(p, p) for p in providers])
    ax.set_ylabel("Mean Score")
    ax.set_title("Provider-Level Fairness Profile with 95% Bootstrap CI",
                 fontweight="bold")
    ax.legend(title="Metric", loc="upper right")
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    _save_fig(fig, "fig2_provider_profile")


# ── FIG 3 / 4 / 6 helper: version trajectory per provider family ─────────────

def _make_trajectory(df, metric, stem):
    """
    One subplot per provider family (≥2 models in data), showing the
    chronological trajectory of *metric* mean ± 95% bootstrap CI band.
    Layout: ncols=3, figsize=(7, 6).
    """
    available = set(df["model"].unique())
    families  = [fam for fam, models in VERSION_ORDERING.items()
                 if sum(1 for m in models if m in available) >= 2]

    if not families:
        print(f"  No families with ≥2 models for {metric} — skipping {stem}")
        return

    ncols = 3
    nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 8), squeeze=False)
    fig.subplots_adjust(hspace=0.40, wspace=0.35)

    for ax_idx, family in enumerate(families):
        row, col = divmod(ax_idx, ncols)
        ax       = axes[row][col]
        ordered  = [m for m in VERSION_ORDERING[family] if m in available]
        color    = FAMILY_COLORS[family]
        x        = np.arange(len(ordered))
        means, lo_abs, hi_abs = [], [], []

        for model in ordered:
            vals = df[df["model"] == model][metric].dropna().values
            mn, lo, hi = _bootstrap_ci(vals)
            means.append(mn)
            lo_abs.append(lo)
            hi_abs.append(hi)

        means  = np.array(means,  dtype=float)
        lo_abs = np.array(lo_abs, dtype=float)
        hi_abs = np.array(hi_abs, dtype=float)

        # CI shaded band and dotted boundary lines
        ax.fill_between(x, lo_abs, hi_abs, color=color, alpha=0.15, zorder=2)
        ax.plot(x, lo_abs, linestyle=":", color=color, alpha=0.50, linewidth=0.9, zorder=2)
        ax.plot(x, hi_abs, linestyle=":", color=color, alpha=0.50, linewidth=0.9, zorder=2)
        # Main trajectory line
        ax.plot(x, means, marker="o", linewidth=2.0, markersize=6,
                color=color, alpha=0.95, zorder=3)

        # Per-point mean annotation above each marker
        for xi, yi in zip(x, means):
            if not np.isnan(yi):
                ax.annotate(f"{yi:.3f}", xy=(xi, yi),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=10, color=color)

        tick_labels = [SHORT_LABEL.get(m, m.split("/")[-1][:8]) for m in ordered]
        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=10)
        ax.set_title(FAMILY_DISPLAY[family], fontweight="bold", color=color, fontsize=13)
        ax.set_ylabel(METRIC_DISPLAY[metric], fontsize=11)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.45, len(ordered) - 0.55)

        # Net drift annotation (first → last version)
        if len(means) >= 2 and not (np.isnan(means[0]) or np.isnan(means[-1])):
            drift = means[-1] - means[0]
            sign  = "▲" if drift >  0.001 else ("▼" if drift < -0.001 else "—")
            clr   = "#c0392b" if drift > 0.001 else ("#27ae60" if drift < -0.001 else "#7f8c8d")
            ax.text(0.97, 0.04, f"{sign} {drift:+.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=10, color=clr, fontweight="bold")

    # Hide unused subplot slots
    for ax_idx in range(len(families), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    metric_label = METRIC_DISPLAY[metric]
    fig.suptitle(
        f"Version-Level Progression of {metric_label} Scores "
        "with 95% Bootstrap CI",
        fontsize=13, fontweight="bold",
    )
    _save_fig(fig, stem)


# ── FIG 5: Benchmark Sensitivity Heatmap ─────────────────────────────────────

def make_fig5(df):
    """
    Heatmap of std dev of per-model means, per benchmark × metric.
    Measures how much models disagree on each benchmark+metric combination.
    figsize=(7, 4).
    """
    datasets = sorted(df["dataset"].unique())
    labels   = [METRIC_DISPLAY[m] for m in METRIC_COLS]
    matrix   = np.zeros((len(datasets), len(METRIC_COLS)))

    for di, ds in enumerate(datasets):
        ds_df = df[df["dataset"] == ds]
        for mi, metric in enumerate(METRIC_COLS):
            model_means = ds_df.groupby("model")[metric].mean().values
            matrix[di, mi] = float(np.std(model_means)) if len(model_means) > 1 else 0.0

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   vmin=0, vmax=matrix.max() or 1)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=10)

    thresh = matrix.max() * 0.55
    for di in range(len(datasets)):
        for mi in range(len(METRIC_COLS)):
            val = matrix[di, mi]
            ax.text(mi, di, f"{val:.4f}", ha="center", va="center",
                    fontsize=10,
                    color="white" if val > thresh else "black")

    cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03)
    cb.set_label("Std Dev of per-model means", fontsize=11)
    cb.ax.tick_params(labelsize=10)

    ax.set_title("Benchmark Sensitivity: Inter-Model Variability\n"
                 "(Std Dev of model-level means per benchmark × metric)",
                 fontweight="bold", fontsize=13)

    fig.tight_layout()
    _save_fig(fig, "fig5_benchmark_sensitivity")


# ── FIG 7: Ablation Comparison Bar Chart ─────────────────────────────────────

def make_fig7():
    """
    Three-panel figure comparing full-run vs. ablation version-to-version drift
    (from ablation_comparison.csv). Horizontal zero line. figsize=(7, 4).
    """
    abl = pd.read_csv(ABLATION_TABLE)
    abl["full_run_drift"] = pd.to_numeric(abl["full_run_drift"], errors="coerce")
    abl["ablation_drift"] = pd.to_numeric(abl["ablation_drift"], errors="coerce")
    abl = abl.dropna(subset=["full_run_drift", "ablation_drift"]).copy()

    METRICS_ORDER = ["Sentiment", "Toxicity", "Stereotype"]
    METRIC_NOTES  = {
        "Sentiment":  "↑ = more positive",
        "Toxicity":   "↓ = less toxic",
        "Stereotype": "↓ = less stereotypical",
    }
    FAMILIES_ORDER = abl["provider_family"].unique().tolist()
    FULL_COLOR     = "#2c7bb6"
    ABL_COLOR      = "#d7191c"
    FAMILY_FACE    = {fam: FAMILY_COLORS[fam]
                      for fam in FAMILIES_ORDER if fam in FAMILY_COLORS}

    bar_w = 0.30
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.40)

    for mi, metric in enumerate(METRICS_ORDER):
        ax          = axes[mi]
        metric_data = abl[abl["metric"] == metric]
        full_vals, abl_vals, valid_fams = [], [], []

        for fam in FAMILIES_ORDER:
            row = metric_data[metric_data["provider_family"] == fam]
            if row.empty:
                continue
            full_vals.append(float(row["full_run_drift"].iloc[0]))
            abl_vals.append(float(row["ablation_drift"].iloc[0]))
            valid_fams.append(fam)

        x = np.arange(len(valid_fams))

        for fi, (fam, fv, av) in enumerate(zip(valid_fams, full_vals, abl_vals)):
            fc = FAMILY_FACE.get(fam, "#888888")
            ax.bar(x[fi] - bar_w / 2, fv, bar_w,
                   color=fc, alpha=0.85, edgecolor=FULL_COLOR, linewidth=1.6)
            ax.bar(x[fi] + bar_w / 2, av, bar_w,
                   color=fc, alpha=0.45, hatch="///",
                   edgecolor=ABL_COLOR, linewidth=1.6)
            # Compact value labels rotated 90°
            for xpos, val in [(x[fi] - bar_w / 2, fv), (x[fi] + bar_w / 2, av)]:
                pad = (abs(val) + 1e-4) * 0.12
                va  = "bottom" if val >= 0 else "top"
                ax.text(xpos, val + (pad if val >= 0 else -pad), f"{val:+.3f}",
                        ha="center", va=va, fontsize=7, rotation=90)

        ax.axhline(0, color="black", linewidth=0.9, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([FAMILY_DISPLAY[f] for f in valid_fams],
                           rotation=45, ha="right", fontsize=10)
        ax.set_title(f"{metric}\n{METRIC_NOTES[metric]}", fontweight="bold", fontsize=12)
        ax.set_ylabel("Mean drift" if mi == 0 else "", fontsize=11)
        # Expand y-limits so rotated annotations don't clip
        ylo, yhi = ax.get_ylim()
        pad = (yhi - ylo) * 0.22
        ax.set_ylim(ylo - pad, yhi + pad)

    # Legend: run type + family colours
    run_handles = [
        Patch(facecolor="#aaaaaa", edgecolor=FULL_COLOR, linewidth=1.6,
              label="Full run (7 benchmarks)"),
        Patch(facecolor="#aaaaaa", edgecolor=ABL_COLOR, linewidth=1.6,
              hatch="///", alpha=0.55,
              label="Ablation (5 benchmarks)"),
    ]
    fam_handles = [
        Patch(facecolor=FAMILY_FACE.get(f, "#888888"), alpha=0.85,
              label=FAMILY_DISPLAY[f])
        for f in FAMILIES_ORDER if f in FAMILY_FACE
    ]
    fig.legend(handles=run_handles + fam_handles, loc="lower center",
               ncol=3, fontsize=9, framealpha=0.9,
               bbox_to_anchor=(0.5, -0.12))

    fig.suptitle("Drift Conclusions Are Robust to Benchmark Exclusion",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _save_fig(fig, "fig7_ablation_comparison")


# ── FIG 8: Inter-Metric Correlation Scatter Plots ────────────────────────────

def make_fig8(df):
    """
    Three scatter plots (1×3) for every pairwise metric combination.
    OLS regression + 95% prediction band; Pearson r and p-value annotated.
    Provider color-coded.  figsize=(7, 4).
    """
    PAIRS = [
        ("sentiment_score",  "toxicity_score",   "Sentiment",  "Toxicity"),
        ("sentiment_score",  "stereotype_score", "Sentiment",  "Stereotype"),
        ("toxicity_score",   "stereotype_score", "Toxicity",   "Stereotype"),
    ]
    default_color = "#7f7f7f"
    JITTER        = 0.012
    rng_jit       = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(top=0.84, wspace=0.35)

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, PAIRS):
        sub   = df[[xcol, ycol, "provider"]].dropna()
        x_raw = sub[xcol].values
        y_raw = sub[ycol].values

        xj = x_raw + rng_jit.uniform(-JITTER, JITTER, size=len(x_raw))
        yj = y_raw + rng_jit.uniform(-JITTER, JITTER, size=len(y_raw))

        for prov, idx in sub.groupby("provider").groups.items():
            c  = PROVIDER_COLORS.get(prov, default_color)
            xi = np.where(sub.index.isin(idx))[0]
            ax.scatter(xj[xi], yj[xi], c=c, alpha=0.18, s=10,
                       linewidths=0, label=prov, rasterized=True)

        slope, intercept, r_val, p_val, se = stats.linregress(x_raw, y_raw)
        x_fit   = np.linspace(x_raw.min(), x_raw.max(), 200)
        y_fit   = slope * x_fit + intercept
        n       = len(x_raw)
        x_bar   = x_raw.mean()
        se_pred = se * np.sqrt(1 / n + (x_fit - x_bar) ** 2 /
                               np.sum((x_raw - x_bar) ** 2))

        ax.plot(x_fit, y_fit, color="#111111", linewidth=1.8, zorder=5)
        ax.fill_between(x_fit,
                        y_fit - 1.96 * se_pred,
                        y_fit + 1.96 * se_pred,
                        color="#111111", alpha=0.10, zorder=4)

        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        sig   = "*" if p_val < 0.05 else " (n.s.)"
        sign  = "−" if r_val < 0 else ""   # proper minus sign
        ax.text(0.04, 0.96,
                f"r = {sign}{abs(r_val):.3f}{sig}\n{p_str}\nn = {n:,}",
                transform=ax.transAxes, va="top", ha="left", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.35", fc="white",
                          ec="#bbbbbb", alpha=0.92))

        ax.set_xlabel(f"{xlabel} Score", fontsize=11)
        ax.set_ylabel(f"{ylabel} Score", fontsize=11)
        ax.tick_params(labelsize=11)
        ax.set_title(f"{xlabel} vs. {ylabel}", fontweight="bold", fontsize=11)

    # Provider colour legend on last panel
    handles = [
        plt.scatter([], [], c=PROVIDER_COLORS.get(p, default_color),
                    s=30, alpha=0.80, label=p.capitalize())
        for p in PROVIDER_COLORS if p in df["provider"].unique()
    ]
    axes[-1].legend(handles=handles, title="Provider",
                    loc="lower right", fontsize=11, title_fontsize=11,
                    framealpha=0.9)

    fig.suptitle(
        "Pairwise Inter-Metric Correlations  "
        "(jitter ±0.012; regression on raw values)",
        fontsize=12, fontweight="bold", y=0.98,
    )
    _save_fig(fig, "fig8_metric_correlations")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Loading {RESULTS_FILE} ...")
    df = pd.read_csv(RESULTS_FILE)
    if "is_error" in df.columns:
        df = df[df["is_error"] == False].copy()
    for col in METRIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"  {len(df):,} rows | {df['model'].nunique()} models | "
          f"{df['provider'].nunique()} providers | "
          f"{df['dataset'].nunique()} datasets")

    FIGURES = [
        ("fig1_aggregate_drift",
         lambda: make_fig1(df)),
        ("fig2_provider_profile",
         lambda: make_fig2(df)),
        ("fig3_sentiment_trajectory",
         lambda: _make_trajectory(df, "sentiment_score", "fig3_sentiment_trajectory")),
        ("fig4_toxicity_trajectory",
         lambda: _make_trajectory(df, "toxicity_score", "fig4_toxicity_trajectory")),
        ("fig5_benchmark_sensitivity",
         lambda: make_fig5(df)),
        ("fig6_stereotype_trajectory",
         lambda: _make_trajectory(df, "stereotype_score", "fig6_stereotype_trajectory")),
        ("fig7_ablation_comparison",
         lambda: make_fig7()),
        ("fig8_metric_correlations",
         lambda: make_fig8(df)),
    ]

    failed = []
    for fig_name, fn in FIGURES:
        print(f"\n[{fig_name}] ...")
        try:
            fn()
        except Exception as exc:
            import traceback
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            failed.append((fig_name, str(exc)))

    print("\n" + "=" * 60)
    print("examples/plots/ contents:")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        size  = os.path.getsize(fpath)
        tag   = "PDF" if fname.endswith(".pdf") else "PNG"
        print(f"  [{tag}] {fname:<52s}  {size:>10,} bytes")

    if failed:
        print(f"\nFAILED figures ({len(failed)}):")
        for name, err in failed:
            print(f"  {name}: {err}")
    else:
        print("\nAll 8 figures generated successfully (PDF + PNG each).")
