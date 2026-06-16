"""
Cheap repository validation that does not require API keys or network access.

This is intentionally narrower than a full benchmark run. It catches syntax
errors and verifies that the pipeline core can be imported.
"""

from pathlib import Path
import py_compile
import sys


PYTHON_FILES = [
    "main.py",
    "src/__init__.py",
    "src/config.py",
    "src/data_loader.py",
    "src/fairness_metrics.py",
    "src/metrics.py",
    "src/model_interface.py",
    "src/visualization.py",
    "src/pipeline/__init__.py",
    "src/pipeline/execution.py",
    "src/pipeline/filtering.py",
    "src/pipeline/io.py",
    "src/pipeline/reporting.py",
    "src/pipeline/scoring.py",
]


def main():
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    missing = [path for path in PYTHON_FILES if not (root / path).exists()]
    if missing:
        raise SystemExit(f"Missing expected Python files: {missing}")

    for path in PYTHON_FILES:
        py_compile.compile(str(root / path), doraise=True)

    from src.pipeline.filtering import classify_response
    from src.pipeline.io import _choose_reusable_csv

    assert classify_response("Error: failed")[1] == "api_error"
    assert callable(_choose_reusable_csv)
    print("Repository validation passed.")


if __name__ == "__main__":
    main()
