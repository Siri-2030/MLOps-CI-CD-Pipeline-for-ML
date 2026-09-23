"""
Tests for the model training layer.

Run:
    pytest tests/test_model.py -v
"""

import pytest
from sklearn.ensemble import RandomForestClassifier

from src.config import load_params
from src.data_preprocessing import load_raw_data, preprocess, split_data
from src.train import evaluate_model, train_model


@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def split(params):
    raw_df = load_raw_data(params["data"]["raw_path"], params["data"]["delimiter"])
    processed_df = preprocess(raw_df, params)
    return split_data(processed_df, params)


@pytest.fixture(scope="module")
def trained_model(split, params):
    X_train, X_test, y_train, y_test = split
    return train_model(X_train, y_train, params)


class TestModelTraining:
    def test_model_trains_successfully(self, trained_model):
        assert trained_model is not None
        assert isinstance(trained_model, RandomForestClassifier)

    def test_model_is_fitted(self, trained_model):
        # A fitted RandomForestClassifier exposes estimators_
        assert hasattr(trained_model, "estimators_")
        assert len(trained_model.estimators_) > 0

    def test_model_uses_configured_hyperparameters(self, trained_model, params):
        assert trained_model.n_estimators == params["model"]["n_estimators"]
        assert trained_model.max_depth == params["model"]["max_depth"]


class TestModelPrediction:
    def test_model_can_predict(self, trained_model, split):
        _, X_test, _, _ = split
        predictions = trained_model.predict(X_test)
        assert len(predictions) == len(X_test)

    def test_predictions_are_valid_labels(self, trained_model, split):
        _, X_test, _, _ = split
        predictions = trained_model.predict(X_test)
        assert set(predictions) <= {"low", "medium", "high"}

    def test_predict_single_row(self, trained_model, split):
        _, X_test, _, _ = split
        single_row = X_test.iloc[[0]]
        prediction = trained_model.predict(single_row)
        assert len(prediction) == 1
        assert prediction[0] in {"low", "medium", "high"}


class TestModelEvaluation:
    def test_evaluate_model_returns_expected_keys(self, trained_model, split):
        _, X_test, _, y_test = split
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score"}

    def test_evaluate_model_metrics_in_valid_range(self, trained_model, split):
        _, X_test, _, y_test = split
        metrics = evaluate_model(trained_model, X_test, y_test)
        for name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"

    def test_model_beats_random_baseline(self, trained_model, split):
        # Sanity check: a real trained model should clearly beat
        # random guessing across 3 classes (~0.33 accuracy).
        _, X_test, _, y_test = split
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert metrics["accuracy"] > 0.5