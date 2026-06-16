import json
import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

NUM_PROMPTS = 100
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _format_record(prompt: str, axis: str = None, group: str = None, **metadata) -> Dict[str, Any]:
    """
    Normalize prompt records so downstream code can rely on shared keys.
    """
    record: Dict[str, Any] = {"prompt": prompt}
    if axis is not None:
        record["axis"] = axis
    if group is not None:
        record["group"] = group
    for key, value in metadata.items():
        if value is not None:
            record[key] = value
    return record


# -------------------------
# BOLD
# -------------------------
def _load_bold_fallback(num_prompts: int) -> List[Dict[str, Any]]:
    """
    Fallback loader that downloads the raw JSON file via huggingface_hub
    to avoid dataset builder compatibility issues.
    """
    cache_path = hf_hub_download(repo_id="AlexaAI/bold", filename="train.json")
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    prompts: List[Dict[str, Any]] = []
    for row in data[:num_prompts]:
        ps = row.get("prompts")
        prompt_text = ps[0] if isinstance(ps, list) and ps else ps if isinstance(ps, str) else None
        if not prompt_text:
            continue
        prompts.append(
            _format_record(
                prompt_text,
                axis=row.get("category") or "bold",
                group=row.get("target_group") or row.get("subcategory") or row.get("target"),
                domain=row.get("domain") or row.get("category"),
                topic=row.get("topic"),
                target_group=row.get("target_group"),
            )
        )

    if not prompts:
        raise ValueError("BOLD fallback: no prompts extracted. Check raw dataset format.")
    return prompts


def load_bold(num_prompts: int = NUM_PROMPTS) -> List[Dict[str, Any]]:
    """
    Load BOLD prompts from the official dataset.

    Dataset: AlexaAI/bold (a.k.a AmazonScience/bold)
    Each row has a 'prompts' field which is a list of strings.
    We take the first prompt for each row.
    """
    try:
        ds = load_dataset("AlexaAI/bold", split="train").select(range(num_prompts))
        prompts: List[Dict[str, Any]] = []
        for row in ds:
            domain = row.get("domain") or row.get("category")
            axis = row.get("category") or "bold"
            group = row.get("target_group") or row.get("subcategory") or row.get("target")
            ps = row.get("prompts")
            if isinstance(ps, list) and len(ps) > 0:
                prompt_text = ps[0]
            elif isinstance(ps, str):
                prompt_text = ps
            else:
                prompt_text = None

            if prompt_text:
                prompts.append(
                    _format_record(
                        prompt_text,
                        axis=axis,
                        group=group,
                        domain=domain,
                        topic=row.get("topic"),
                        target_group=row.get("target_group"),
                    )
                )

        if not prompts:
            raise ValueError("BOLD: no prompts extracted. Check dataset schema.")
        return prompts
    except ValueError as e:
        if "Feature type 'List' not found" in str(e):
            print("Falling back to raw JSON loader for BOLD due to datasets version incompatibility.")
            return _load_bold_fallback(num_prompts)
        raise


# -------------------------
# StereoSet (intersentence)
# -------------------------
def load_stereoset(num_prompts: int = NUM_PROMPTS) -> List[Dict[str, Any]]:
    """
    McGill-NLP/stereoset, intersentence split.
    We use the 'context' field as the prompt.
    """
    ds = load_dataset("McGill-NLP/stereoset", "intersentence", split="validation")
    ds = ds.select(range(num_prompts))

    prompts: List[Dict[str, Any]] = []
    for row in ds:
        ctx = row.get("context")
        if ctx:
            prompts.append(
                _format_record(
                    ctx,
                    axis=row.get("bias_type") or "stereoset",
                    group=row.get("target"),
                    bias_type=row.get("bias_type"),
                    target=row.get("target"),
                    stereotype=row.get("sent_more"),
                    anti_stereotype=row.get("sent_less"),
                )
            )

    if not prompts:
        raise ValueError("StereoSet: no 'context' fields found.")

    return prompts


# -------------------------
# BBQ (Age category)
# -------------------------
def load_bbq(num_prompts: int = NUM_PROMPTS) -> List[Dict[str, Any]]:
    """
    heegyu/bbq, Age category.
    We combine context + question into one prompt.
    """
    ds = load_dataset("heegyu/bbq", "Age", split="test")
    ds = ds.select(range(num_prompts))

    prompts: List[Dict[str, Any]] = []
    for row in ds:
        context = row.get("context", "")
        question = row.get("question", "")
        text = (context + " " + question).strip()
        if text:
            axis = row.get("category") or "bbq"
            group = (
                row.get("protected_value")
                or row.get("protected_group")
                or row.get("attribute")
                or axis
            )
            prompts.append(
                _format_record(
                    text,
                    axis=axis,
                    group=group,
                    category=row.get("category"),
                    protected_value=row.get("protected_value"),
                    question=row.get("question"),
                    answer_info=row.get("answer_info"),
                    is_ambiguous=row.get("is_ambiguous"),
                )
            )

    if not prompts:
        raise ValueError("BBQ: no (context, question) prompts built.")

    return prompts


# -------------------------
# CrowS-Pairs
# -------------------------
def load_crows_pairs(num_prompts: int = NUM_PROMPTS) -> List[Dict[str, Any]]:
    """
    Load CrowS-Pairs locally to avoid fragile remote script execution.
    """
    csv_path = DATA_DIR / "crows_pairs_anonymized.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CrowS-Pairs CSV not found at {csv_path}. "
            "Download it from https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv "
            "and place it in the data/ directory."
        )

    df = pd.read_csv(csv_path)
    if "sent_more" not in df.columns:
        raise ValueError("CrowS-Pairs CSV missing 'sent_more' column.")

    prompts: List[Dict[str, Any]] = []
    for _, row in df.head(num_prompts).iterrows():
        sent = row["sent_more"]
        if isinstance(sent, str) and sent.strip():
            prompts.append(
                _format_record(
                    sent,
                    axis=row.get("bias_type") or "crows_pairs",
                    group=row.get("bias_target") or row.get("category"),
                    bias_type=row.get("bias_type"),
                    bias_target=row.get("bias_target"),
                    sent_less=row.get("sent_less"),
                )
            )

    if not prompts:
        raise ValueError("CrowS-Pairs: no valid prompts extracted from CSV.")

    return prompts


# -------------------------
# RealToxicityPrompts
# -------------------------
def load_real_toxicity_prompts(num_prompts: int = NUM_PROMPTS) -> List[Dict[str, Any]]:
    """
    allenai/real-toxicity-prompts, train split.
    Each row has a nested 'prompt' dict with a 'text' field.
    """
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    ds = ds.select(range(num_prompts))

    prompts: List[Dict[str, Any]] = []
    for row in ds:
        prompt_obj = row.get("prompt", {})
        text = prompt_obj.get("text")
        if text:
            prompts.append(
                _format_record(
                    text,
                    axis=prompt_obj.get("category") or "toxicity",
                    group=prompt_obj.get("subcategory") or row.get("challenger"),
                    prompt_id=row.get("id"),
                    source=row.get("source"),
                    prompt_toxicity=prompt_obj.get("toxicity"),
                )
            )

    if not prompts:
        raise ValueError("RealToxicityPrompts: no prompt['text'] fields found.")

    return prompts


# -------------------------
# HolisticBias
# -------------------------
def load_holistic_bias(num_prompts: int = NUM_PROMPTS) -> List[Dict[str, Any]]:
    """
    fairnlp/holistic-bias with sentences.csv.
    We use 'sentence' when available, otherwise 'text'.
    """
    ds = load_dataset(
        "fairnlp/holistic-bias",
        data_files=["sentences.csv"],
        split="train",
    )
    ds = ds.select(range(num_prompts))

    prompts: List[Dict[str, Any]] = []
    for row in ds:
        sentence = row.get("sentence") or row.get("text")
        if sentence:
            axis = row.get("axis") or row.get("category") or "holistic_bias"
            group = row.get("descriptor") or row.get("group") or row.get("target")
            prompts.append(
                _format_record(
                    sentence,
                    axis=axis,
                    group=group,
                    persona=row.get("persona"),
                    template_id=row.get("template_id"),
                    axis_source=row.get("axis"),
                )
            )

    if not prompts:
        raise ValueError("HolisticBias: neither 'sentence' nor 'text' found.")

    return prompts


# -------------------------
# WinoBias
# -------------------------
def load_winobias(num_prompts: int = NUM_PROMPTS) -> List[Dict[str, Any]]:
    """
    Elfsong/Wino_Bias:
    Use the 'input' field from the train split and attach gender/polarity metadata.
    """
    ds = load_dataset("Elfsong/Wino_Bias", split="train")
    n = min(num_prompts, len(ds))
    prompts: List[Dict[str, Any]] = []

    for i in range(n):
        row = ds[i]
        prompt_text = row.get("input")
        if not prompt_text:
            continue
        prompts.append(
            _format_record(
                prompt_text,
                axis="gender_identity",
                group=row.get("gender"),
                polarity=row.get("polarity"),
                reference=row.get("reference"),
                wb_type=row.get("type"),
            )
        )

    if len(prompts) < n:
        raise ValueError(
            f"WinoBias: only collected {len(prompts)} prompts; expected {n}."
        )

    return prompts


# -------------------------
# Combined loader
# -------------------------
def load_all_datasets(num_prompts: int = NUM_PROMPTS):
    """
    Returns a dict:
        {
            "bold": [...100 prompts...],
            "stereoset": [...],
            ...
        }
    All prompts are REAL dataset text; if a dataset fails, an exception is raised.
    """
    return {
        "bold": load_bold(num_prompts),
        "stereoset": load_stereoset(num_prompts),
        "bbq": load_bbq(num_prompts),
        "crows_pairs": load_crows_pairs(num_prompts),
        "realtoxicityprompts": load_real_toxicity_prompts(num_prompts),
        "holistic_bias": load_holistic_bias(num_prompts),
        "winobias": load_winobias(num_prompts),
    }
