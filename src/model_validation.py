"""
Model validation gate.

Compares the metrics from the most recent training run
(models/metrics.json) against the minimum thresholds configured
in params.yaml. Exits with status code 1 if the model does not
meet the bar, which CI uses to block Docker build / deployment.

Run locally:
    python -m src.model_validation

This mirrors GitHub Actions CI Stage 8 (Model Validation).
"""

import json
import sys
from pathlib import Path

from src.config import load_params

METRICS_PATH = Path("models/metrics.json")


def fail(message: str) -> None:
    print(f"[VALIDATION FAILED] {message}")
    sys.exit(1)


def load_metrics(path: Path = METRICS_PATH) -> dict:
    if not path.exists():
        fail(
            f"Metrics file not found at {path}. "
            "Run training (python -m src.train) before validation."
        )
    with open(path, "r") as f:
        return json.load(f)


def validate(metrics: dict, params: dict) -> bool:
    min_accuracy = params["validation"]["min_accuracy"]
    min_f1 = params["validation"]["min_f1"]

    accuracy = metrics.get("accuracy")
    f1 = metrics.get("f1_score")

    if accuracy is None or f1 is None:
        fail(f"Metrics file missing required keys. Found: {list(metrics.keys())}")

    print(f"Required:  accuracy >= {min_accuracy}, f1_score >= {min_f1}")
    print(f"Measured:  accuracy = {accuracy:.4f}, f1_score = {f1:.4f}")

    passed = True

    if accuracy < min_accuracy:
        print(f"  [FAIL] accuracy {accuracy:.4f} < required {min_accuracy}")
        passed = False
    else:
        print(f"  [PASS] accuracy {accuracy:.4f} >= required {min_accuracy}")

    if f1 < min_f1:
        print(f"  [FAIL] f1_score {f1:.4f} < required {min_f1}")
        passed = False
    else:
        print(f"  [PASS] f1_score {f1:.4f} >= required {min_f1}")

    return passed


def main() -> None:
    params = load_params()
    metrics = load_metrics()

    print("Running model validation gate...")
    passed = validate(metrics, params)

    if not passed:
        fail("Model does not meet minimum performance thresholds. Deployment blocked.")

    print("\n[VALIDATION PASSED] Model meets minimum performance thresholds.")
    sys.exit(0)


if __name__ == "__main__":
    main()