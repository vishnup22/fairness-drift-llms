from datetime import datetime

import pandas as pd
from tqdm import tqdm

from src.config import ALL_MODELS, HF_MODELS, MAX_TOKENS, PROMPTS_PER_MODEL
from src.data_loader import load_all_datasets
from src.pipeline.io import save_raw_results


def make_run_id(exclude_datasets=None, temperature=0.0, prefix=None):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix:
        run_id = f"{prefix}_{run_id}"
    if exclude_datasets:
        excl_tag = "excl_" + "_".join(sorted(d.strip() for d in exclude_datasets if d.strip()))
        run_id = f"{run_id}_{excl_tag}"
    if temperature != 0.0:
        run_id = f"{run_id}_temp{temperature}"
    return run_id


def select_model_sets(include_hf_models=True, hf_only=False, provider_filter=None, model_filter=None):
    if hf_only:
        model_sets = HF_MODELS.copy()
        print("Models: Llama 3.1, Llama 3.2, Gemma")
    else:
        model_sets = ALL_MODELS.copy()
        if include_hf_models:
            model_sets.update(HF_MODELS)
            print("Models: OpenAI (ChatGPT), Claude, Gemini, Llama 3.1, Llama 3.2, Gemma")
        else:
            print("Models: OpenAI (ChatGPT), Claude, Gemini")

    if provider_filter:
        keep = {p.strip() for p in provider_filter if p.strip()}
        model_sets = {provider: models for provider, models in model_sets.items() if provider in keep}
        if not model_sets:
            raise ValueError(f"Provider filter {keep} yielded no providers.")
        print(f"Provider filter active -> {list(model_sets.keys())}")

    if model_filter:
        selected_models = {m.strip() for m in model_filter if m.strip()}
        filtered = {}
        for provider, models in model_sets.items():
            keep = [m for m in models if m in selected_models]
            if keep:
                filtered[provider] = keep
        model_sets = filtered
        if not model_sets:
            raise ValueError(f"Model filter {selected_models} yielded no models.")
        print(f"Model filter active -> {model_sets}")

    return model_sets


def load_selected_datasets(dataset_filter=None, exclude_datasets=None):
    print("Loading benchmark datasets:")
    datasets = load_all_datasets()

    if dataset_filter:
        keep = {d.strip() for d in dataset_filter if d.strip()}
        datasets = {name: prompts for name, prompts in datasets.items() if name in keep}
        if not datasets:
            raise ValueError(f"Dataset filter {keep} yielded no datasets.")
        print(f"Dataset filter active -> {list(datasets.keys())}")

    if exclude_datasets:
        exclude_set = {d.strip() for d in exclude_datasets if d.strip()}
        excluded_found = sorted(name for name in datasets if name in exclude_set)
        unrecognised = exclude_set - set(datasets.keys())
        datasets = {name: prompts for name, prompts in datasets.items() if name not in exclude_set}
        if not datasets:
            raise ValueError(
                f"--exclude-datasets removed all loaded datasets ({sorted(exclude_set)}). Nothing to run."
            )
        print(f"Excluded datasets: {excluded_found} -> remaining: {sorted(datasets.keys())}")
        if unrecognised:
            print(f"  Warning: these names in --exclude-datasets were not loaded: {sorted(unrecognised)}")

    print(f"Loaded {len(datasets)} datasets\n")
    return datasets


def apply_dataframe_filters(df, dataset_filter=None, exclude_datasets=None, provider_filter=None, model_filter=None):
    df = df.copy()

    if dataset_filter:
        keep_ds = {d.strip() for d in dataset_filter if d.strip()}
        df = df[df["dataset"].isin(keep_ds)].copy()
        if df.empty:
            raise ValueError(f"Dataset filter {keep_ds} yielded no rows in loaded results.")
        print(f"  Dataset filter -> {sorted(df['dataset'].unique())}")

    if exclude_datasets:
        exclude_set = {d.strip() for d in exclude_datasets if d.strip()}
        df = df[~df["dataset"].isin(exclude_set)].copy()
        if df.empty:
            raise ValueError("--exclude-datasets removed all loaded results.")
        print(f"  Excluded datasets: {sorted(exclude_set)} -> remaining: {sorted(df['dataset'].unique())}")

    if provider_filter:
        keep_prov = {p.strip() for p in provider_filter if p.strip()}
        df = df[df["provider"].isin(keep_prov)].copy()
        if df.empty:
            raise ValueError(f"Provider filter {keep_prov} yielded no rows in loaded results.")
        print(f"  Provider filter -> {sorted(df['provider'].unique())}")

    if model_filter:
        keep_mod = {m.strip() for m in model_filter if m.strip()}
        df = df[df["model"].isin(keep_mod)].copy()
        if df.empty:
            raise ValueError(f"Model filter {keep_mod} yielded no rows in loaded results.")
        print(f"  Model filter -> {sorted(df['model'].unique())}")

    return df


def run_inference(model_sets, datasets, temperature=0.0):
    from src.model_interface import query_model

    results = []

    for dataset_name, prompts in datasets.items():
        print(f"TESTING ON DATASET: {dataset_name.upper()}")
        dataset_prompts = prompts[:min(len(prompts), PROMPTS_PER_MODEL // len(datasets))]

        for provider, models in model_sets.items():
            print(f"\nProvider: {provider.upper()}")

            for model in models:
                print(f"  Running {model}...")

                for i, prompt_item in enumerate(tqdm(dataset_prompts, desc=f"  {model}")):
                    if isinstance(prompt_item, dict):
                        prompt_text = prompt_item.get("prompt", "")
                        metadata = {k: v for k, v in prompt_item.items() if k != "prompt"}
                    else:
                        prompt_text = str(prompt_item)
                        metadata = {}

                    if not prompt_text:
                        continue

                    response = query_model(provider, model, prompt_text, MAX_TOKENS, temperature=temperature)

                    record = {
                        "timestamp": datetime.now().isoformat(),
                        "provider": provider,
                        "model": model,
                        "dataset": dataset_name,
                        "prompt_id": i,
                        "prompt": prompt_text,
                        "response": response,
                    }
                    record.update(metadata)
                    results.append(record)

                print(f" Completed {model}\n")

    return pd.DataFrame(results)


def execute_fresh_run(
    include_hf_models=True,
    hf_only=False,
    dataset_filter=None,
    provider_filter=None,
    model_filter=None,
    exclude_datasets=None,
    temperature=0.0,
):
    model_sets = select_model_sets(
        include_hf_models=include_hf_models,
        hf_only=hf_only,
        provider_filter=provider_filter,
        model_filter=model_filter,
    )
    datasets = load_selected_datasets(
        dataset_filter=dataset_filter,
        exclude_datasets=exclude_datasets,
    )
    df_raw = run_inference(model_sets, datasets, temperature=temperature)
    run_id = make_run_id(exclude_datasets=exclude_datasets, temperature=temperature)
    save_raw_results(df_raw, run_id)
    return df_raw, run_id
