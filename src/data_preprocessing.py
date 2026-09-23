"""
Data preprocessing.

Loads the raw Wine Quality CSV, converts the numeric `quality` score
into low/medium/high class labels (per params.yaml bucket config),
and produces train/test feature and label sets.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_params


def load_raw_data(path: str, delimiter: str = ";") -> pd.DataFrame:
    """Load the raw dataset from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found at {path}")
    return pd.read_csv(path, sep=delimiter)


def bucket_quality(quality: int, buckets: dict) -> str:
    """Map a numeric quality score to a low/medium/high label."""
    for label, (low, high) in buckets.items():
        if low <= quality <= high:
            return label
    raise ValueError(f"Quality value {quality} does not fall into any configured bucket")


def preprocess(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Apply the quality bucketing to produce the classification target."""
    target_col = params["target"]["column"]
    buckets = params["target"]["buckets"]

    df = df.copy()
    df["quality_class"] = df[target_col].apply(lambda q: bucket_quality(q, buckets))
    return df


def split_data(df: pd.DataFrame, params: dict):
    """Split processed data into train/test feature and label sets."""
    feature_cols = [c for c in df.columns if c not in ("quality", "quality_class")]
    X = df[feature_cols]
    y = df["quality_class"]

    return train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )


def run_preprocessing_pipeline():
    """Full pipeline: load raw -> bucket target -> save processed CSV."""
    params = load_params()

    raw_df = load_raw_data(params["data"]["raw_path"], params["data"]["delimiter"])
    processed_df = preprocess(raw_df, params)

    out_path = Path(params["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed_df.to_csv(out_path, index=False)

    print(f"Processed dataset saved to: {out_path}")
    print(f"Class distribution:\n{processed_df['quality_class'].value_counts()}")

    return processed_df


if __name__ == "__main__":
    run_preprocessing_pipeline()