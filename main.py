from src.pipeline.execution import (
    apply_dataframe_filters,
    execute_fresh_run,
    make_run_id,
)
from src.pipeline.filtering import apply_response_filtering
from src.pipeline.io import load_existing_results, save_metrics_results
from src.pipeline.reporting import generate_artifacts, print_run_summary
from src.pipeline.scoring import score_clean_results


def run_benchmark(
    include_hf_models=True,
    hf_only=False,
    dataset_filter=None,
    provider_filter=None,
    model_filter=None,
    exclude_datasets=None,
    temperature=0.0,
    reuse_results=False,
):
    """
    Run the benchmark as an explicit stage pipeline:
      1. load or generate raw results
      2. apply row-level filters
      3. compute/reuse metrics
      4. save scored results
      5. generate visualizations/tables
      6. print summary
    """
    if reuse_results:
        print("\n--reuse-results: skipping API inference, loading existing raw CSVs.")
        df_raw = load_existing_results(include_hf_models=include_hf_models, hf_only=hf_only)
        df_raw = apply_dataframe_filters(
            df_raw,
            dataset_filter=dataset_filter,
            exclude_datasets=exclude_datasets,
            provider_filter=provider_filter,
            model_filter=model_filter,
        )
        run_id = make_run_id(exclude_datasets=exclude_datasets, temperature=temperature, prefix="reuse")
    else:
        df_raw, run_id = execute_fresh_run(
            include_hf_models=include_hf_models,
            hf_only=hf_only,
            dataset_filter=dataset_filter,
            provider_filter=provider_filter,
            model_filter=model_filter,
            exclude_datasets=exclude_datasets,
            temperature=temperature,
        )

    df_clean, total_filtered, filter_rate = apply_response_filtering(df_raw)
    df_clean, metrics = score_clean_results(df_clean, total_filtered, filter_rate)

    save_metrics_results(df_clean, run_id)
    generate_artifacts(df_clean, metrics)
    print_run_summary(df_raw, df_clean, metrics, total_filtered)


def run_hf_models_benchmark():
    """Run benchmark on Hugging Face models only."""
    return run_benchmark(include_hf_models=True, hf_only=True)


def parse_args(args):
    options = {
        "hf_only": "--hf-only" in args,
        "include_hf_models": "--no-hf" not in args,
        "reuse_results": "--reuse-results" in args,
        "dataset_filter": None,
        "provider_filter": None,
        "model_filter": None,
        "exclude_datasets": None,
        "temperature": 0.0,
    }

    for arg in args:
        if arg.startswith("--dataset="):
            options["dataset_filter"] = arg.split("=", 1)[1].split(",")
        elif arg.startswith("--provider="):
            options["provider_filter"] = arg.split("=", 1)[1].split(",")
        elif arg.startswith("--model="):
            options["model_filter"] = arg.split("=", 1)[1].split(",")
        elif arg.startswith("--exclude-datasets="):
            options["exclude_datasets"] = arg.split("=", 1)[1].split(",")
        elif arg.startswith("--temperature="):
            options["temperature"] = float(arg.split("=", 1)[1])

    return options


if __name__ == "__main__":
    import sys

    run_benchmark(**parse_args(sys.argv[1:]))
