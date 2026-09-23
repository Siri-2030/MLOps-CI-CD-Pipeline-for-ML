"""
Tests for the model validation gate.

Run:
    pytest tests/test_validation.py -v
"""

import pytest

from src.model_validation import validate


@pytest.fixture
def params():
    return {
        "validation": {
            "min_accuracy": 0.70,
            "min_f1": 0.65,
        }
    }


class TestValidationGate:
    def test_passes_when_metrics_meet_thresholds(self, params):
        metrics = {"accuracy": 0.75, "f1_score": 0.70}
        assert validate(metrics, params) is True

    def test_passes_when_metrics_exactly_equal_thresholds(self, params):
        metrics = {"accuracy": 0.70, "f1_score": 0.65}
        assert validate(metrics, params) is True

    def test_fails_when_accuracy_below_threshold(self, params):
        metrics = {"accuracy": 0.50, "f1_score": 0.70}
        assert validate(metrics, params) is False

    def test_fails_when_f1_below_threshold(self, params):
        metrics = {"accuracy": 0.80, "f1_score": 0.40}
        assert validate(metrics, params) is False

    def test_fails_when_both_metrics_below_threshold(self, params):
        metrics = {"accuracy": 0.40, "f1_score": 0.30}
        assert validate(metrics, params) is False

    def test_fails_when_impossibly_high_threshold_set(self):
        strict_params = {"validation": {"min_accuracy": 0.99, "min_f1": 0.99}}
        metrics = {"accuracy": 0.75, "f1_score": 0.70}
        assert validate(metrics, strict_params) is False