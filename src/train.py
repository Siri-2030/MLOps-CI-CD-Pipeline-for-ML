"""
Model training.

Loads the processed Wine Quality dataset, trains a Random Forest
classifier on the low/medium/high quality_class target, evaluates
it, and saves the trained model to disk.

Run locally:
    python -m src.train
"""

import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import load_params
from src.data_preprocessing import load_raw_data, preprocess, split_data

MODEL_PATH = Path("models/model.pkl")
METRICS_PATH = Path("models/metrics.json")


def train_model(X_train, y_train, params: dict) -> RandomForestClassifier:
    model_cfg = params["model"]
    clf = RandomForestClassifier(
        n_estimators=model_cfg["n_estimators"],
        max_depth=model_cfg["max_depth"],
        random_state=model_cfg["random_state"],
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="macro", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }

    print("\nClassification report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred, labels=["low", "medium", "high"]))

    return metrics


def run_training_pipeline():
    params = load_params()

    raw_df = load_raw_data(params["data"]["raw_path"], params["data"]["delimiter"])
    processed_df = preprocess(raw_df, params)
    X_train, X_test, y_train, y_test = split_data(processed_df, params)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    model = train_model(X_train, y_train, params)
    metrics = evaluate_model(model, X_test, y_test)

    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

    # Save metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    return model, metrics


if __name__ == "__main__":
    run_training_pipeline()