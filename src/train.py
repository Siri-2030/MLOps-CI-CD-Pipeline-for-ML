"""
Model training.

Loads the processed Wine Quality dataset, trains a Random Forest
classifier on the low/medium/high quality_class target, evaluates
it, logs the run to MLflow (params, metrics, model artifact), and
saves the trained model to disk for the FastAPI app to load.

Run locally:
    python -m src.train

View MLflow results:
    mlflow ui --backend-store-uri mlruns
    (then open http://127.0.0.1:5000)
"""

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
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

    mlflow.set_tracking_uri(params["mlflow"]["tracking_uri"])
    mlflow.set_experiment(params["mlflow"]["experiment_name"])

    raw_df = load_raw_data(params["data"]["raw_path"], params["data"]["delimiter"])
    processed_df = preprocess(raw_df, params)
    X_train, X_test, y_train, y_test = split_data(processed_df, params)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    with mlflow.start_run():
        # --- Log parameters ---
        model_cfg = params["model"]
        mlflow.log_param("model_type", model_cfg["type"])
        mlflow.log_param("n_estimators", model_cfg["n_estimators"])
        mlflow.log_param("max_depth", model_cfg["max_depth"])
        mlflow.log_param("random_state", model_cfg["random_state"])
        mlflow.log_param("test_size", params["data"]["test_size"])

        # --- Train ---
        model = train_model(X_train, y_train, params)
        metrics = evaluate_model(model, X_test, y_test)

        print("\nMetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        # --- Log metrics ---
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # --- Log model artifact to MLflow ---
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID: {run_id}")

    # --- Save model + metrics locally too, for the FastAPI app ---
    # (FastAPI loads directly from models/model.pkl rather than
    # querying MLflow at request time, to keep serving simple/fast)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")

    metrics["mlflow_run_id"] = run_id
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {METRICS_PATH}")

    return model, metrics


if __name__ == "__main__":
    run_training_pipeline()