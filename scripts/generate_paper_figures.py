"""
Generate three publication-ready figures for the EMNLP paper.

Outputs (all to examples/plots/):
  fig_version_progression_stereotype_mean.pdf
  fig_ablation_comparison.pdf
  fig_metric_correlations.pdf
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy import stats

# ── shared style (mirrors visualization.py) ───────────────────────────────────
import matplotlib as mpl
mpl.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

PLOTS_DIR = "plots"

# VERSION_ORDERING from config.py — defines chronological order within families
VERSION_ORDERING = {
    "openai": [
        "gpt-4-turbo", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini",
    ],
    "claude_sonnet": [
        "claude-sonnet-4-20250514", "claude-sonnet-4-5-20250929",
    ],
    "claude_opus": [
        "claude-opus-4-1-20250805",
    ],
    "gemini": [
        "gemini-2.0-flash", "gemini-2.5-flash",
        "gemini-2.5-flash-lite", "gemini-2.5-pro",
    ],
    "llama31": [
        "meta-llama/Llama-3.1-8B-Instruct",
        "meta-llama/Llama-3.1-70B-Instruct",
        "meta-llama/Llama-3.1-405B-Instruct",
    ],
    "llama32": [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ],
    "gemma": [
        "google/gemma-2-2b-it", "google/gemma-2-9b-it",
    ],
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

# Colours matching the existing plots' palette (tab10-based)
FAMILY_COLORS = {
    "openai":        "#1f77b4",  # blue
    "claude_sonnet": "#ff7f0e",  # orange
    "claude_opus":   "#d62728",  # red
    "gemini":        "#2ca02c",  # green
    "llama31":       "#9467bd",  # purple
    "llama32":       "#8c564b",  # brown
    "gemma":         "#e377c2",  # pink
}

SHORT_LABEL = {
    "gpt-4-turbo":                           "GPT-4-\nturbo",
    "gpt-4o":                                "GPT-4o",
    "gpt-4o-mini":                           "GPT-4o\nmini",
    "gpt-4.1":                               "GPT-4.1",
    "gpt-4.1-mini":                          "GPT-4.1\nmini",
    "claude-sonnet-4-20250514":              "Sonnet-4\n(May)",
    "claude-sonnet-4-5-20250929":            "Sonnet-4.5\n(Sep)",
    "claude-opus-4-1-20250805":              "Opus-4.1\n(Aug)",
    "gemini-2.0-flash":                      "2.0\nFlash",
    "gemini-2.5-flash":                      "2.5\nFlash",
    "gemini-2.5-flash-lite":                 "2.5 Flash\nLite",
    "gemini-2.5-pro":                        "2.5\nPro",
    "meta-llama/Llama-3.1-8B-Instruct":     "8B",
    "meta-llama/Llama-3.1-70B-Instruct":    "70B",
    "meta-llama/Llama-3.1-405B-Instruct":   "405B",
    "meta-llama/Llama-3.2-1B-Instruct":     "1B",
    "meta-llama/Llama-3.2-3B-Instruct":     "3B",
    "google/gemma-2-2b-it":                  "2B",
    "google/gemma-2-9b-it":                  "9B",
}


def _bootstrap_ci(values, n=1000, seed=42, alpha=0.05):
    """Return (mean, ci_lo, ci_hi) via bootstrap resampling (n=1000, seed=42)."""
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean()
                     for _ in range(n)])
    return float(vals.mean()), float(np.percentile(boot, 100 * alpha / 2)), \
           float(np.percentile(boot, 100 * (1 - alpha / 2)))


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Stereotype Score Version Progression
# ─────────────────────────────────────────────────────────────────────────────

def make_fig1(df):
    """
    One subplot per provider family, showing stereotype score mean ± 95% CI
    across successive model versions. Matches the style of the existing
    sentiment and toxicity trajectory plots.
    """
    METRIC = "stereotype_score"
    available = set(df["model"].unique())

    # Only families with ≥ 2 models in the data
    families = [
        fam for fam, models in VERSION_ORDERING.items()
        if sum(1 for m in models if m in available) >= 2
    ]

    ncols = min(4, len(families))
    nrows = (len(families) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(4.2 * ncols, 3.8 * nrows),
                             squeeze=False)

    for ax_idx, family in enumerate(families):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]

        ordered = [m for m in VERSION_ORDERING[family] if m in available]
        color   = FAMILY_COLORS[family]
        x       = np.arange(len(ordered))
        means, lo, hi = [], [], []

        for model in ordered:
            vals = df[df["model"] == model][METRIC].dropna().values
            m, l, h = _bootstrap_ci(vals)
            means.append(m)
            lo.append(l)
            hi.append(h)

        means = np.array(means, dtype=float)
        lo    = np.array(lo,    dtype=float)
        hi    = np.array(hi,    dtype=float)

        # CI shaded band
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
        # CI caps (dashed)
        ax.plot(x, lo, linestyle=":", color=color, alpha=0.45, linewidth=0.9)
        ax.plot(x, hi, linestyle=":", color=color, alpha=0.45, linewidth=0.9)
        # Main trajectory
        ax.plot(x, means, marker="o", linewidth=2.2, markersize=7,
                color=color, alpha=0.9, zorder=3)

        # Annotate each point with its mean value
        for xi, yi in zip(x, means):
            if not np.isnan(yi):
                ax.annotate(f"{yi:.3f}", xy=(xi, yi),
                            xytext=(0, 7), textcoords="offset points",
                            ha="center", fontsize=6.5, color=color)

        labels = [SHORT_LABEL.get(m, m.split("/")[-1][:10]) for m in ordered]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7.5)
        ax.set_title(FAMILY_DISPLAY[family], fontsize=10, fontweight="bold",
                     color=color)
        ax.set_ylabel("Stereotype Score", fontsize=9)
        ax.set_ylim(bottom=0)
        ax.set_xlim(-0.4, len(ordered) - 0.6)

        # Mark drift direction with subtle arrow between first and last
        if len(means) >= 2 and not (np.isnan(means[0]) or np.isnan(means[-1])):
            drift = means[-1] - means[0]
            sign  = "▲" if drift > 0.001 else ("▼" if drift < -0.001 else "—")
            clr   = "#c0392b" if drift > 0.001 else ("#27ae60" if drift < -0.001 else "#7f8c8d")
            ax.text(0.97, 0.05, f"{sign} {drift:+.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, color=clr, fontweight="bold")

    # Hide unused axes
    for ax_idx in range(len(families), nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        "Version-Level Progression of Stereotype Scores\n"
        "with 95% Bootstrap Confidence Intervals",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    out = f"{PLOTS_DIR}/fig_version_progression_stereotype_mean.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Ablation Comparison Bar Chart
# ─────────────────────────────────────────────────────────────────────────────

def make_fig2():
    """
    Side-by-side bars comparing mean drift in the full 7-benchmark run vs.
    the 5-benchmark ablation. Three panels (one per metric) with independent
    y-axes so that very different drift magnitudes across metrics don't
    compress any panel into illegibility.
    """
    abl = pd.read_csv("outputs/tables/ablation_comparison.csv")
    abl["full_run_drift"] = pd.to_numeric(abl["full_run_drift"], errors="coerce")
    abl["ablation_drift"] = pd.to_numeric(abl["ablation_drift"], errors="coerce")
    abl = abl.dropna(subset=["full_run_drift", "ablation_drift"]).copy()

    METRICS_ORDER  = ["Sentiment", "Toxicity", "Stereotype"]
    METRIC_NOTES   = {
        "Sentiment":  "higher = more positive",
        "Toxicity":   "negative = less toxic (better)",
        "Stereotype": "negative = less stereotypical (better)",
    }
    FAMILIES_ORDER = abl["provider_family"].unique().tolist()

    FULL_COLOR = "#2c7bb6"
    ABL_COLOR  = "#d7191c"

    # One colour per family for bar face so families are distinguishable
    FAMILY_FACE = {
        fam: FAMILY_COLORS[fam] for fam in FAMILIES_ORDER if fam in FAMILY_COLORS
    }

    bar_w = 0.32
    xs    = np.arange(len(FAMILIES_ORDER))

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    for mi, metric in enumerate(METRICS_ORDER):
        ax = axes[mi]
        metric_data = abl[abl["metric"] == metric]

        full_vals = []
        abl_vals  = []
        valid_fams = []

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
            # Full run bar — family colour, solid edge
            ax.bar(x[fi] - bar_w / 2, fv, bar_w,
                   color=fc, alpha=0.85,
                   edgecolor=FULL_COLOR, linewidth=1.8,
                   label=None)
            # Ablation bar — family colour, hatched, dashed edge
            ax.bar(x[fi] + bar_w / 2, av, bar_w,
                   color=fc, alpha=0.45, hatch="///",
                   edgecolor=ABL_COLOR, linewidth=1.8,
                   label=None)

            # Annotate value above/below each bar
            for xpos, val in [(x[fi] - bar_w / 2, fv), (x[fi] + bar_w / 2, av)]:
                offset = 0.0008 if val >= 0 else -0.0008
                va     = "bottom" if val >= 0 else "top"
                ax.text(xpos, val + offset, f"{val:+.4f}",
                        ha="center", va=va, fontsize=6.2, rotation=90)

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [FAMILY_DISPLAY[f] for f in valid_fams],
            rotation=30, ha="right", fontsize=7.5,
        )
        ax.set_title(f"{metric}\n({METRIC_NOTES[metric]})",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean version-to-version drift" if mi == 0 else "",
                      fontsize=9)
        # Expand y limits slightly so annotations don't clip
        ylo, yhi = ax.get_ylim()
        pad = (yhi - ylo) * 0.18
        ax.set_ylim(ylo - pad, yhi + pad)

    # ── shared legend ─────────────────────────────────────────────────────────
    run_legend = [
        Patch(facecolor="#aaaaaa", edgecolor=FULL_COLOR, linewidth=1.8,
              label="Full run (7 benchmarks) — solid border"),
        Patch(facecolor="#aaaaaa", edgecolor=ABL_COLOR,  linewidth=1.8,
              hatch="///", alpha=0.55,
              label="Ablation (5 benchmarks) — hatched, dashed border"),
    ]
    fam_legend = [
        Patch(facecolor=FAMILY_FACE.get(f, "#888888"), alpha=0.85,
              label=FAMILY_DISPLAY[f])
        for f in FAMILIES_ORDER if f in FAMILY_FACE
    ]
    fig.legend(
        handles=run_legend + fam_legend,
        loc="lower center", ncol=3,
        fontsize=7.8, framealpha=0.9,
        bbox_to_anchor=(0.5, -0.08),
    )

    fig.suptitle(
        "Ablation Comparison: Mean Version-to-Version Drift\n"
        "Full 7-Benchmark Run vs. 5-Benchmark Ablation (excl. StereoSet & CrowS-Pairs)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out = f"{PLOTS_DIR}/fig_ablation_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Inter-Metric Scatter / Correlation Plots
# ─────────────────────────────────────────────────────────────────────────────

def make_fig3(df):
    """
    Three scatter plots (1 x 3) for the three pairwise metric combinations,
    with OLS regression lines and Pearson r / p-value annotations.

    Scores are discrete-valued (classifier outputs at fixed levels), so we add
    small independent jitter on both axes to reveal density rather than showing
    solid horizontal bands.
    """
    PAIRS = [
        ("sentiment_score",  "toxicity_score",   "Sentiment",  "Toxicity"),
        ("sentiment_score",  "stereotype_score",  "Sentiment",  "Stereotype"),
        ("toxicity_score",   "stereotype_score",  "Toxicity",   "Stereotype"),
    ]

    PROV_COLORS = {
        "claude":  "#ff7f0e",
        "openai":  "#1f77b4",
        "gemini":  "#2ca02c",
        "llama31": "#9467bd",
        "llama32": "#8c564b",
        "gemma":   "#e377c2",
    }
    default_color = "#7f7f7f"

    # Jitter scale — small enough to preserve cluster identity, large enough to
    # show density structure within each discrete level.
    JITTER = 0.012
    rng_jit = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, PAIRS):
        sub = df[[xcol, ycol, "provider"]].dropna()
        x_raw = sub[xcol].values
        y_raw = sub[ycol].values

        # Jittered coordinates (for display only — regression uses raw values)
        xj = x_raw + rng_jit.uniform(-JITTER, JITTER, size=len(x_raw))
        yj = y_raw + rng_jit.uniform(-JITTER, JITTER, size=len(y_raw))

        for prov, idx in sub.groupby("provider").groups.items():
            c  = PROV_COLORS.get(prov, default_color)
            xi = np.where(sub.index.isin(idx))[0]
            ax.scatter(xj[xi], yj[xi],
                       c=c, alpha=0.20, s=8, linewidths=0,
                       label=prov, rasterized=True)

        # OLS on raw (un-jittered) values
        slope, intercept, r_val, p_val, se = stats.linregress(x_raw, y_raw)
        x_fit   = np.linspace(x_raw.min(), x_raw.max(), 200)
        y_fit   = slope * x_fit + intercept
        n       = len(x_raw)
        x_bar   = x_raw.mean()
        se_pred = se * np.sqrt(1/n + (x_fit - x_bar)**2 /
                               np.sum((x_raw - x_bar)**2))

        ax.plot(x_fit, y_fit, color="#111111", linewidth=2.0, zorder=5,
                label="_nolegend_")
        ax.fill_between(x_fit,
                         y_fit - 1.96 * se_pred,
                         y_fit + 1.96 * se_pred,
                         color="#111111", alpha=0.12, zorder=4)

        # Annotation box — r with correct sign
        p_str = "p < 0.001" if p_val < 0.001 else f"p = {p_val:.3f}"
        sig   = "*" if p_val < 0.05 else " (n.s.)"
        sign  = "-" if r_val < 0 else ""
        ax.text(0.04, 0.96,
                f"$r = {sign}{abs(r_val):.3f}${sig}\n{p_str}\n$n = {n:,}$",
                transform=ax.transAxes,
                va="top", ha="left", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#cccccc", alpha=0.9))

        ax.set_xlabel(f"{xlabel} Score", fontsize=10)
        ax.set_ylabel(f"{ylabel} Score", fontsize=10)
        # Clean title — no LaTeX escapes needed
        ax.set_title(f"{xlabel} vs. {ylabel}", fontsize=11, fontweight="bold")

    # Provider legend on the last panel
    handles = [
        plt.scatter([], [], c=PROV_COLORS.get(p, default_color),
                    s=35, alpha=0.75, label=p.capitalize())
        for p in PROV_COLORS
        if p in df["provider"].unique()
    ]
    axes[-1].legend(handles=handles, title="Provider",
                    loc="lower right", fontsize=7.5, title_fontsize=8,
                    framealpha=0.9)

    fig.suptitle(
        "Pairwise Inter-Metric Correlations Across All Models and Benchmarks\n"
        r"(points jittered $\pm$0.012 to reveal density; regression on raw values)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()
    out = f"{PLOTS_DIR}/fig_metric_correlations.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading outputs/results/results_with_metrics_20251115_035107.csv ...")
    df = pd.read_csv("outputs/results/results_with_metrics_20251115_035107.csv")
    if "is_error" in df.columns:
        df = df[df["is_error"] == False].copy()
    for col in ["sentiment_score", "toxicity_score", "stereotype_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"  {len(df):,} rows | {df['model'].nunique()} models | "
          f"{df['provider'].nunique()} providers | "
          f"{df['dataset'].nunique()} datasets")

    print("\nFigure 1 — Stereotype version progression …")
    make_fig1(df)

    print("Figure 2 — Ablation comparison bar chart …")
    make_fig2()

    print("Figure 3 — Inter-metric correlation scatter plots …")
    make_fig3(df)

    print("\nDone. All figures saved to examples/plots/")
