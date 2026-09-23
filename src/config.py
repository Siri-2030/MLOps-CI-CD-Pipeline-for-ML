"""
Configuration loader.

Reads params.yaml so that training, validation, and preprocessing
code share a single source of truth for parameters, instead of
hard-coding values across multiple files (see Section 21 of spec).
"""

from pathlib import Path
from typing import Any

import yaml

PARAMS_PATH = Path(__file__).resolve().parent.parent / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict[str, Any]:
    """Load and return the full params.yaml as a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found at {path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        params = yaml.safe_load(f)
    return params


if __name__ == "__main__":
    # Quick manual check: python src/config.py
    params = load_params()
    print("Loaded params.yaml successfully:")
    print(params)
# Trigger CI/CD pipeline
