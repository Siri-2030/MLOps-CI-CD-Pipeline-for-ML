"""
Dataset validation script.

Checks that the raw Wine Quality dataset exists and meets basic
quality requirements before it is used for training.

Run locally:
    python scripts/validate_data.py

This same script is called by the GitHub Actions CI pipeline
(Stage 6 - Dataset Validation) so that a bad dataset blocks the
pipeline before any time is spent training a model on it.
"""

import sys
from pathlib import Path

import pandas as pd

DATA_PATH = Path("data/raw/winequality-red.csv")

EXPECTED_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality",
]

MIN_ROWS = 1000
VALID_QUALITY_RANGE = (0, 10)


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def main() -> None:
    print(f"Validating dataset at: {DATA_PATH}")

    # 1. Dataset exists
    if not DATA_PATH.exists():
        fail(f"Dataset not found at {DATA_PATH}")

    # 2. Dataset can be loaded (semicolon-delimited, per UCI format)
    try:
        df = pd.read_csv(DATA_PATH, sep=";")
    except Exception as e:
        fail(f"Failed to load dataset: {e}")

    print(f"Loaded dataset with shape: {df.shape}")

    # 3. Required columns exist
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        fail(f"Missing expected columns: {missing_cols}")
    print("All expected columns present.")

    # 4. Minimum dataset size
    if len(df) < MIN_ROWS:
        fail(f"Dataset has only {len(df)} rows, expected at least {MIN_ROWS}")
    print(f"Row count OK: {len(df)} rows (minimum {MIN_ROWS}).")

    # 5. No unexpected missing values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        fail(f"Found {total_nulls} missing values:\n{null_counts[null_counts > 0]}")
    print("No missing values found.")

    # 6. Correct data types (all columns should be numeric)
    non_numeric = df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric:
        fail(f"Non-numeric columns found (expected all numeric): {non_numeric}")
    print("All columns are numeric.")

    # 7. Valid target values
    min_q, max_q = VALID_QUALITY_RANGE
    invalid_quality = df[(df["quality"] < min_q) | (df["quality"] > max_q)]
    if len(invalid_quality) > 0:
        fail(f"Found {len(invalid_quality)} rows with quality outside {VALID_QUALITY_RANGE}")
    print(f"All 'quality' values within valid range {VALID_QUALITY_RANGE}.")

    print("\n[PASS] Dataset validation successful.")


if __name__ == "__main__":
    main()