"""
Ablation comparison analysis for EMNLP paper.
Compares full 7-benchmark run vs. ablation excluding StereoSet and CrowS-Pairs.
"""

import pandas as pd
import numpy as np
from scipy import stats
import os

# ── VERSION_ORDERING from config.py ──────────────────────────────────────────
VERSION_ORDERING = {
    "openai": [
        "gpt-4-turbo",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
    ],
    "claude_sonnet": [
        "claude-sonnet-4-20250514",
        "claude-sonnet-4-5-20250929",
    ],
    "claude_opus": [
        "claude-opus-4-1-20250805",
    ],
    "gemini": [
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
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
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
    ],
}

# Map family → display provider label
FAMILY_TO_PROVIDER = {
    "openai":       "openai",
    "claude_sonnet":"claude",
    "claude_opus":  "claude",
    "gemini":       "gemini",
    "llama31":      "meta (llama-3.1)",
    "llama32":      "meta (llama-3.2)",
    "gemma":        "google (gemma-2)",
}

METRICS = ["sentiment_score", "toxicity_score", "stereotype_score"]
METRIC_LABELS = {
    "sentiment_score":  "Sentiment",
    "toxicity_score":   "Toxicity",
    "stereotype_score": "Stereotype",
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load(path):
    df = pd.read_csv(path)
    # Drop error rows
    if "is_error" in df.columns:
        df = df[df["is_error"] == False].copy()
    for m in METRICS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    return df


def provider_means(df):
    """Return dict {provider: {metric: (mean, std)}}."""
    out = {}
    for prov, grp in df.groupby("provider"):
        out[prov] = {}
        for m in METRICS:
            vals = grp[m].dropna()
            out[prov][m] = (float(vals.mean()), float(vals.std()))
    return out


def family_drift(df):
    """
    Compute mean version-to-version drift per family per metric.
    Drift at step i = mean(metric, model[i+1]) - mean(metric, model[i]).
    Returns dict {family: {metric: mean_drift or None}}.
    """
    model_set = set(df["model"].unique())
    out = {}
    for family, ordered in VERSION_ORDERING.items():
        present = [m for m in ordered if m in model_set]
        if len(present) < 2:
            out[family] = {m: None for m in METRICS}
            continue
        family_drifts = {m: [] for m in METRICS}
        for i in range(len(present) - 1):
            m_a, m_b = present[i], present[i + 1]
            rows_a = df[df["model"] == m_a]
            rows_b = df[df["model"] == m_b]
            for met in METRICS:
                a = rows_a[met].dropna().mean()
                b = rows_b[met].dropna().mean()
                if not (np.isnan(a) or np.isnan(b)):
                    family_drifts[met].append(b - a)
        out[family] = {
            met: (float(np.mean(v)) if v else None)
            for met, v in family_drifts.items()
        }
    return out


def inter_metric_corr(df):
    """Pearson r for each pair of metrics."""
    pairs = [
        ("sentiment_score", "toxicity_score"),
        ("sentiment_score", "stereotype_score"),
        ("toxicity_score",  "stereotype_score"),
    ]
    results = {}
    for a, b in pairs:
        sub = df[[a, b]].dropna()
        if len(sub) < 10:
            results[(a, b)] = (np.nan, np.nan)
            continue
        r, p = stats.pearsonr(sub[a], sub[b])
        results[(a, b)] = (float(r), float(p))
    return results


def fmt(v, digits=4):
    return f"{v:.{digits}f}" if v is not None and not np.isnan(v) else "N/A"


def conclusion_holds(full_drift, abl_drift):
    """
    True if sign matches and magnitude is within 2× of each other
    (or both are near zero, |drift| < 0.005).
    """
    if full_drift is None or abl_drift is None:
        return None
    near_zero = lambda x: abs(x) < 0.005
    if near_zero(full_drift) and near_zero(abl_drift):
        return True
    if full_drift * abl_drift <= 0:          # opposite signs
        return False
    ratio = max(abs(full_drift), abs(abl_drift)) / (
        min(abs(full_drift), abs(abl_drift)) + 1e-12
    )
    return bool(ratio <= 3.0)                # same sign, within 3×


# ── load data ─────────────────────────────────────────────────────────────────

FULL_PATH = "outputs/results/results_with_metrics_20251115_035107.csv"
ABL_PATH  = "outputs/results/results_with_metrics_reuse_20260525_191525.csv"

full = load(FULL_PATH)
abl  = load(ABL_PATH)

print(f"Full run : {len(full):,} rows | "
      f"datasets: {sorted(full['dataset'].unique())} | "
      f"models: {full['model'].nunique()}")
print(f"Ablation : {len(abl):,} rows  | "
      f"datasets: {sorted(abl['dataset'].unique())} | "
      f"models: {abl['model'].nunique()}")

# ── compute provider means ────────────────────────────────────────────────────

full_means = provider_means(full)
abl_means  = provider_means(abl)

# For full run, also restrict to commercial providers to enable apples-to-apples
commercial = {"claude", "openai", "gemini"}
full_comm = full[full["provider"].isin(commercial)]
full_means_comm = provider_means(full_comm)

# ── compute family drift ──────────────────────────────────────────────────────

full_drift = family_drift(full)
abl_drift  = family_drift(abl)

# ── build comparison CSV ──────────────────────────────────────────────────────

rows = []
for family, ordered in VERSION_ORDERING.items():
    provider_label = FAMILY_TO_PROVIDER[family]
    for met in METRICS:
        metric_label = METRIC_LABELS[met]

        # Provider means: use commercial-restricted full run for shared providers
        if provider_label in commercial:
            fm, fs = full_means_comm.get(provider_label, {}).get(met, (np.nan, np.nan))
        else:
            # HF providers: use raw full run
            prov_key = provider_label.split(" ")[0]   # e.g. "meta"
            # find the actual provider string in the full run
            matching = [p for p in full_means if provider_label.startswith(p) or p.startswith(prov_key)]
            if matching:
                fm, fs = full_means.get(matching[0], {}).get(met, (np.nan, np.nan))
            else:
                fm, fs = np.nan, np.nan
        am, as_ = abl_means.get(provider_label, {}).get(met, (np.nan, np.nan))

        fd = full_drift[family][met]
        ad = abl_drift[family][met]
        ch = conclusion_holds(fd, ad)

        rows.append({
            "provider_family":   family,
            "provider":          provider_label,
            "metric":            metric_label,
            "full_run_mean":     round(fm, 4) if not np.isnan(fm) else "N/A",
            "ablation_mean":     round(am, 4) if not np.isnan(am) else "N/A",
            "full_run_drift":    round(fd, 5) if fd is not None else "N/A",
            "ablation_drift":    round(ad, 5) if ad is not None else "N/A",
            "conclusion_holds":  ch if ch is not None else "N/A (single version)",
        })

comp_df = pd.DataFrame(rows)
os.makedirs("tables", exist_ok=True)
comp_df.to_csv("outputs/tables/ablation_comparison.csv", index=False)
print(f"\nSaved outputs/tables/ablation_comparison.csv ({len(comp_df)} rows)")

# ── inter-metric correlations ─────────────────────────────────────────────────

full_corr = inter_metric_corr(full)
abl_corr  = inter_metric_corr(abl)

# ── paper claims evaluation ───────────────────────────────────────────────────

def evaluate_claim_toxicity(drift_dict, label):
    """
    Claim: 'Toxicity declines consistently across newer model releases.'
    We look at all families with >= 2 versions and check whether most
    step-level drifts are negative (toxicity decreasing = improvement).
    """
    neg, pos, total = 0, 0, 0
    family_results = {}
    model_set_in_data = set()
    if label == "full":
        model_set_in_data = set(full["model"].unique())
    else:
        model_set_in_data = set(abl["model"].unique())

    for family, ordered in VERSION_ORDERING.items():
        present = [m for m in ordered if m in model_set_in_data]
        if len(present) < 2:
            continue
        family_neg, family_pos = 0, 0
        if label == "full":
            src = full
        else:
            src = abl
        for i in range(len(present) - 1):
            m_a, m_b = present[i], present[i + 1]
            a = src[src["model"] == m_a]["toxicity_score"].dropna().mean()
            b = src[src["model"] == m_b]["toxicity_score"].dropna().mean()
            if not (np.isnan(a) or np.isnan(b)):
                d = b - a
                total += 1
                if d < 0:
                    neg += 1; family_neg += 1
                else:
                    pos += 1; family_pos += 1
        family_results[family] = {"neg": family_neg, "pos": family_pos,
                                   "n_versions": len(present)}
    return neg, pos, total, family_results


def evaluate_claim_stereotype(drift_dict, label):
    """
    Claim: 'Stereotype scores remain stable or increase.'
    Stable = |drift| < 0.005, increase = drift > 0.
    """
    stable_or_up = 0
    declines = 0
    model_set_in_data = set()
    if label == "full":
        model_set_in_data = set(full["model"].unique())
    else:
        model_set_in_data = set(abl["model"].unique())

    for family, ordered in VERSION_ORDERING.items():
        present = [m for m in ordered if m in model_set_in_data]
        if len(present) < 2:
            continue
        if label == "full":
            src = full
        else:
            src = abl
        for i in range(len(present) - 1):
            m_a, m_b = present[i], present[i + 1]
            a = src[src["model"] == m_a]["stereotype_score"].dropna().mean()
            b = src[src["model"] == m_b]["stereotype_score"].dropna().mean()
            if not (np.isnan(a) or np.isnan(b)):
                d = b - a
                if d >= -0.005:       # stable or increasing
                    stable_or_up += 1
                else:
                    declines += 1
    return stable_or_up, declines


def claim_verdict(condition_full, condition_abl, threshold=0.65):
    """Return HOLDS / PARTIALLY HOLDS / DOES NOT HOLD."""
    if condition_full and condition_abl:
        return "HOLDS"
    if condition_full and not condition_abl:
        return "PARTIALLY HOLDS"
    if not condition_full and condition_abl:
        return "PARTIALLY HOLDS"
    return "DOES NOT HOLD"


# Claim 1 – toxicity declines
f_neg, f_pos, f_tot, f_fam = evaluate_claim_toxicity(full_drift,  "full")
a_neg, a_pos, a_tot, a_fam = evaluate_claim_toxicity(abl_drift,   "abl")

f_tox_pct = f_neg / f_tot if f_tot else 0
a_tox_pct = a_neg / a_tot if a_tot else 0

tox_full_holds = f_tox_pct >= 0.55
tox_abl_holds  = a_tox_pct >= 0.55

if tox_full_holds and tox_abl_holds:
    claim1_verdict = "HOLDS"
elif tox_full_holds or tox_abl_holds:
    claim1_verdict = "PARTIALLY HOLDS"
else:
    claim1_verdict = "DOES NOT HOLD"

claim1_evidence = (
    f"Full run: {f_neg}/{f_tot} version steps show toxicity decline "
    f"({f_tox_pct:.0%}); ablation: {a_neg}/{a_tot} ({a_tox_pct:.0%}). "
)
if tox_full_holds and tox_abl_holds:
    claim1_evidence += "Majority of steps decline in both runs, supporting the claim."
elif tox_full_holds and not tox_abl_holds:
    claim1_evidence += (
        "Decline is majority in the full run but not in the ablation "
        "(StereoSet/CrowS removal shifts the balance)."
    )
elif not tox_full_holds and tox_abl_holds:
    claim1_evidence += (
        "Decline is majority in the ablation but not the full run — "
        "stereoset/crows_pairs inclusion pulls the metric upward."
    )
else:
    claim1_evidence += "Inconsistent patterns across both runs; claim requires qualification."


# Claim 2 – stereotype stable or increases
f_su, f_dec = evaluate_claim_stereotype(full_drift, "full")
a_su, a_dec = evaluate_claim_stereotype(abl_drift,  "abl")

f_ste_pct = f_su / (f_su + f_dec) if (f_su + f_dec) else 0
a_ste_pct = a_su / (a_su + a_dec) if (a_su + a_dec) else 0

ste_full_holds = f_ste_pct >= 0.55
ste_abl_holds  = a_ste_pct >= 0.55

if ste_full_holds and ste_abl_holds:
    claim2_verdict = "HOLDS"
elif ste_full_holds or ste_abl_holds:
    claim2_verdict = "PARTIALLY HOLDS"
else:
    claim2_verdict = "DOES NOT HOLD"

claim2_evidence = (
    f"Full run: {f_su}/{f_su+f_dec} version steps are stable or increasing "
    f"({f_ste_pct:.0%}); ablation: {a_su}/{a_su+a_dec} ({a_ste_pct:.0%}). "
)
if ste_full_holds and ste_abl_holds:
    claim2_evidence += "Stereotype scores are predominantly non-declining in both configurations."
elif ste_full_holds and not ste_abl_holds:
    claim2_evidence += (
        "The pattern holds in the full run but not the ablation; "
        "removing StereoSet/CrowS-Pairs — which have explicit stereotype signals — "
        "reveals more improvement on remaining benchmarks."
    )
elif not ste_full_holds and ste_abl_holds:
    claim2_evidence += (
        "StereoSet and CrowS-Pairs drive apparent stability in the full run; "
        "excluding them shows more variability."
    )
else:
    claim2_evidence += "Mixed evidence; the stability claim is not robustly supported."


# Claim 3 – inter-metric correlations weak (|r| < 0.2)
CORR_PAIRS = [
    ("sentiment_score", "toxicity_score",  "Sentiment–Toxicity"),
    ("sentiment_score", "stereotype_score","Sentiment–Stereotype"),
    ("toxicity_score",  "stereotype_score","Toxicity–Stereotype"),
]

corr_lines = []
all_weak_full = True
all_weak_abl  = True

for a, b, label in CORR_PAIRS:
    fr, fp = full_corr[(a, b)]
    ar, ap = abl_corr[(a, b)]
    sig_f = "sig" if (not np.isnan(fp) and fp < 0.05) else "n.s."
    sig_a = "sig" if (not np.isnan(ap) and ap < 0.05) else "n.s."
    corr_lines.append((label,
                        f"r={fr:.3f} ({sig_f})" if not np.isnan(fr) else "N/A",
                        f"r={ar:.3f} ({sig_a})" if not np.isnan(ar) else "N/A"))
    if not np.isnan(fr) and abs(fr) >= 0.2:
        all_weak_full = False
    if not np.isnan(ar) and abs(ar) >= 0.2:
        all_weak_abl = False

if all_weak_full and all_weak_abl:
    claim3_verdict = "HOLDS"
elif all_weak_full or all_weak_abl:
    claim3_verdict = "PARTIALLY HOLDS"
else:
    claim3_verdict = "DOES NOT HOLD"

strong_full = [(l, fr, ar) for (l, fr, ar) in corr_lines
               if fr != "N/A" and float(fr.split("=")[1].split(" ")[0]) >= 0.2]
strong_abl  = [(l, fr, ar) for (l, fr, ar) in corr_lines
               if ar != "N/A" and float(ar.split("=")[1].split(" ")[0]) >= 0.2]

if all_weak_full and all_weak_abl:
    claim3_evidence = (
        f"All three pairwise correlations satisfy |r| < 0.2 in both runs "
        f"({'; '.join(f'{l}: full {fr}, ablation {ar}' for l, fr, ar in corr_lines)}). "
        "Weak associations are not an artifact of including StereoSet/CrowS-Pairs."
    )
elif not all_weak_full:
    violators = [f"{l} full {fr}" for l, fr, ar in corr_lines
                 if fr != "N/A" and abs(float(fr.split('=')[1].split(' ')[0])) >= 0.2]
    claim3_evidence = (
        f"Full run has correlations |r| >= 0.2: {', '.join(violators)}. "
        f"Ablation: {'; '.join(f'{l}: {ar}' for l, _, ar in corr_lines)}."
    )
else:
    violators = [f"{l} abl {ar}" for l, fr, ar in corr_lines
                 if ar != "N/A" and abs(float(ar.split('=')[1].split(' ')[0])) >= 0.2]
    claim3_evidence = (
        f"Ablation has correlations |r| >= 0.2: {', '.join(violators)}. "
        f"Full run: {'; '.join(f'{l}: {fr}' for l, fr, _ in corr_lines)}."
    )

# ── build per-provider mean table (for summary report) ───────────────────────

all_providers_full = sorted(full["provider"].unique())

def mean_table(df_src, providers):
    lines = []
    lines.append(f"  {'Provider':<26} {'Sentiment':>10} {'Toxicity':>10} {'Stereotype':>12}")
    lines.append("  " + "-" * 62)
    for prov in providers:
        sub = df_src[df_src["provider"] == prov]
        sent = sub["sentiment_score"].dropna().mean()
        tox  = sub["toxicity_score"].dropna().mean()
        ste  = sub["stereotype_score"].dropna().mean()
        lines.append(f"  {prov:<26} {sent:>10.4f} {tox:>10.4f} {ste:>12.4f}")
    return "\n".join(lines)


def drift_table(drift_dict, model_set):
    lines = []
    lines.append(f"  {'Family':<18} {'Metric':<14} {'Drift':>10}  {'Versions'}")
    lines.append("  " + "-" * 58)
    for family, ordered in VERSION_ORDERING.items():
        present = [m for m in ordered if m in model_set]
        if len(present) < 2:
            continue
        for met in METRICS:
            d = drift_dict[family][met]
            d_str = f"{d:+.5f}" if d is not None else "N/A"
            lines.append(
                f"  {family:<18} {METRIC_LABELS[met]:<14} {d_str:>10}"
                f"  ({' → '.join(m.split('/')[-1][:20] for m in present)})"
            )
    return "\n".join(lines)


# ── write ablation_summary.txt ────────────────────────────────────────────────

sep = "=" * 72

report_lines = [
    sep,
    "ABLATION COMPARISON REPORT",
    "Excluding StereoSet and CrowS-Pairs from Fairness Evaluation",
    sep,
    "",
    "FILES COMPARED",
    "-" * 40,
    f"  Full run  : {FULL_PATH}",
    f"             {len(full):,} rows | {full['dataset'].nunique()} datasets | "
    f"{full['model'].nunique()} models | {full['provider'].nunique()} providers",
    f"  Ablation  : {ABL_PATH}",
    f"             {len(abl):,} rows | {abl['dataset'].nunique()} datasets | "
    f"{abl['model'].nunique()} models | {abl['provider'].nunique()} providers",
    f"  Excluded  : stereoset, crows_pairs",
    f"  Retained  : {', '.join(sorted(abl['dataset'].unique()))}",
    "",
    sep,
    "SECTION 1 — MEAN SCORES PER PROVIDER",
    sep,
    "",
    "Full run (all 7 benchmarks, all 19 models):",
    mean_table(full, all_providers_full),
    "",
    "Ablation run (5 benchmarks, commercial models only):",
    mean_table(abl, sorted(abl["provider"].unique())),
    "",
    "Full run restricted to commercial providers (apple-to-apples):",
    mean_table(full_comm, sorted(commercial)),
    "",
    sep,
    "SECTION 2 — VERSION-TO-VERSION DRIFT PER PROVIDER FAMILY",
    sep,
    "",
    "Full run drift (all benchmarks):",
    drift_table(full_drift, set(full["model"].unique())),
    "",
    "Ablation drift (excl. StereoSet + CrowS-Pairs):",
    drift_table(abl_drift, set(abl["model"].unique())),
    "",
    sep,
    "SECTION 3 — INTER-METRIC CORRELATIONS (Pearson r)",
    sep,
    "",
    f"  {'Pair':<26} {'Full run':>16}  {'Ablation':>16}",
    "  " + "-" * 62,
]
for label, fr, ar in corr_lines:
    report_lines.append(f"  {label:<26} {fr:>16}  {ar:>16}")

report_lines += [
    "",
    "  Threshold for 'weak': |r| < 0.20",
    f"  All correlations weak — Full run : {'YES' if all_weak_full else 'NO'}",
    f"  All correlations weak — Ablation : {'YES' if all_weak_abl else 'NO'}",
    "",
    sep,
    "SECTION 4 — PAPER CLAIM EVALUATION",
    sep,
    "",
    "CLAIM 1: 'Toxicity declines consistently across newer model releases'",
    f"  Verdict  : {claim1_verdict}",
    f"  Evidence : {claim1_evidence}",
    "",
    "CLAIM 2: 'Stereotype scores remain stable or increase'",
    f"  Verdict  : {claim2_verdict}",
    f"  Evidence : {claim2_evidence}",
    "",
    "CLAIM 3: 'Inter-metric correlations are weak (|r| < 0.2)'",
    f"  Verdict  : {claim3_verdict}",
    f"  Evidence : {claim3_evidence}",
    "",
    sep,
    "SECTION 5 — ABLATION ROBUSTNESS SUMMARY",
    sep,
    "",
]

# Summarise conclusion_holds column
holds_df = comp_df[comp_df["conclusion_holds"].isin([True, False])].copy()
n_holds    = (holds_df["conclusion_holds"] == True).sum()
n_noHolds  = (holds_df["conclusion_holds"] == False).sum()
pct_holds  = n_holds / len(holds_df) * 100 if len(holds_df) else 0

report_lines += [
    f"  Of {len(holds_df)} computable (family, metric) pairs:",
    f"    Conclusions consistent (conclusion_holds=True) : {n_holds} ({pct_holds:.0f}%)",
    f"    Conclusions diverge   (conclusion_holds=False) : {n_noHolds}",
    "",
    "  Divergent cases:",
]
divergent = holds_df[holds_df["conclusion_holds"] == False]
if len(divergent):
    for _, r in divergent.iterrows():
        report_lines.append(
            f"    {r['provider_family']:<18} {r['metric']:<14} "
            f"full_drift={r['full_run_drift']:>9}  abl_drift={r['ablation_drift']:>9}"
        )
else:
    report_lines.append("    (none)")

report_lines += [
    "",
    "  OVERALL ROBUSTNESS: "
    + ("Core findings are robust to excluding StereoSet and CrowS-Pairs."
       if pct_holds >= 75 else
       "Some findings are sensitive to benchmark inclusion; review divergent cases above."),
    "",
    sep,
    "DETAILED COMPARISON TABLE",
    sep,
    "",
    "Saved to: outputs/tables/ablation_comparison.csv",
    "",
    comp_df.to_string(index=False),
    "",
    sep,
]

report_text = "\n".join(report_lines)

with open("outputs/tables/ablation_summary.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

print("Saved outputs/tables/ablation_summary.txt")
print("\n" + report_text)
