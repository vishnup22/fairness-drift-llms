import os
from pathlib import Path

import pandas as pd

from src.pipeline.filtering import apply_response_filtering, classify_response
from src.pipeline.io import _choose_reusable_csv


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
