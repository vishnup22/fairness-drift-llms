import os
from pathlib import Path

import pandas as pd

from main import parse_args
from src import config
from src.data_loader import DATA_DIR
from src.pipeline.filtering import apply_response_filtering, classify_response
from src.pipeline.io import _choose_reusable_csv
from src.pipeline.scoring import score_clean_results


def test_classify_response_rejects_invalid_outputs():
    assert classify_response("")[1] == "empty"
    assert classify_response("Error: rate limit")[1] == "api_error"
    assert classify_response("[BLOCKED: SAFETY]")[1] == "policy_blocked"
    assert classify_response("[No text returned]")[1] == "no_text"
    assert classify_response("I'm afraid I can't help with that.")[1] == "refusal"


def test_classify_response_accepts_clean_output():
    should_filter, reason = classify_response("This is a normal model response.")

    assert should_filter is False
    assert reason is None


def test_apply_response_filtering_returns_clean_rows(tmp_path, monkeypatch):
    monkeypatch.setattr("src.pipeline.filtering.TABLES_DIR", str(tmp_path))
    df = pd.DataFrame(
        [
            {
                "provider": "openai",
                "model": "gpt-test",
                "dataset": "bbq",
                "prompt_id": 0,
                "response": "Clean response.",
            },
            {
                "provider": "openai",
                "model": "gpt-test",
                "dataset": "bbq",
                "prompt_id": 1,
                "response": "Error: failed",
            },
        ]
    )

    df_clean, total_filtered, filter_rate = apply_response_filtering(df)

    assert len(df_clean) == 1
    assert total_filtered == 1
    assert filter_rate == 0.5
    assert (tmp_path / "filtering_rate_by_model.csv").exists()
    assert (tmp_path / "filtering_rate_by_benchmark.csv").exists()


def test_choose_reusable_csv_prefers_broadest_file(tmp_path):
    small = tmp_path / "results_with_metrics_reuse_latest.csv"
    large = tmp_path / "results_with_metrics_full_older.csv"

    small.write_text("col\n1\n", encoding="utf-8")
    large.write_text("col\n1\n2\n3\n", encoding="utf-8")

    os.utime(small, (200, 200))
    os.utime(large, (100, 100))

    assert Path(_choose_reusable_csv([str(small), str(large)])).name == large.name


def test_cli_parse_args_supports_reuse_filters_and_temperature():
    args = parse_args([
        "--reuse-results",
        "--no-hf",
        "--dataset=bbq,bold",
        "--provider=openai,claude",
        "--model=gpt-4o,claude-sonnet-4-20250514",
        "--exclude-datasets=stereoset,crows_pairs",
        "--temperature=0.5",
    ])

    assert args["reuse_results"] is True
    assert args["include_hf_models"] is False
    assert args["dataset_filter"] == ["bbq", "bold"]
    assert args["provider_filter"] == ["openai", "claude"]
    assert args["model_filter"] == ["gpt-4o", "claude-sonnet-4-20250514"]
    assert args["exclude_datasets"] == ["stereoset", "crows_pairs"]
    assert args["temperature"] == 0.5


def test_config_output_dirs_point_to_outputs():
    assert config.OUTPUT_DIR == "outputs/results"
    assert config.PLOTS_DIR == "outputs/plots"
    assert config.TABLES_DIR == "outputs/tables"


def test_crows_pairs_data_dir_points_to_data_folder():
    assert DATA_DIR.name == "data"
    assert (DATA_DIR / "crows_pairs_anonymized.csv").exists()


def test_score_clean_results_reuses_metric_columns(monkeypatch):
    monkeypatch.setattr("src.pipeline.scoring.COMPREHENSIVE_METRICS_AVAILABLE", False)
    df = pd.DataFrame(
        [
            {
                "provider": "openai",
                "model": "gpt-test",
                "dataset": "bbq",
                "prompt_id": 0,
                "prompt": "prompt",
                "response": "response",
                "sentiment_score": 0.8,
                "toxicity_score": 0.1,
                "stereotype_score": 0.2,
            }
        ]
    )

    scored, metrics = score_clean_results(df, total_filtered=0, filter_rate=0.0)

    assert scored.equals(df)
    assert metrics["sentiment_method"] == "precomputed"
    assert metrics["mean_sentiment"] == 0.8


def test_model_interface_import_does_not_require_api_keys(monkeypatch):
    import importlib
    import src.model_interface as model_interface

    monkeypatch.setattr(model_interface, "ANTHROPIC_API_KEY", None)
    monkeypatch.setattr(model_interface, "OPENAI_API_KEY", None)
    monkeypatch.setattr(model_interface, "GOOGLE_API_KEY", None)
    importlib.reload(model_interface)

    assert model_interface.query_model("unknown", "model", "prompt").startswith("Error: Unknown provider")
