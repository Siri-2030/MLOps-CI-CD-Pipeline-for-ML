"""
Tests for the data loading and preprocessing layer.

Run:
    pytest tests/test_data.py -v
"""

import pandas as pd
import pytest

from src.config import load_params
from src.data_preprocessing import bucket_quality, load_raw_data, preprocess, split_data


@pytest.fixture(scope="module")
def params():
    return load_params()


@pytest.fixture(scope="module")
def raw_df(params):
    return load_raw_data(params["data"]["raw_path"], params["data"]["delimiter"])


class TestDatasetLoading:
    def test_dataset_file_exists_and_loads(self, raw_df):
        assert raw_df is not None
        assert len(raw_df) > 0

    def test_dataset_has_expected_shape(self, raw_df):
        # 11 features + 1 target column
        assert raw_df.shape[1] == 12

    def test_load_raw_data_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_raw_data("data/raw/does_not_exist.csv", delimiter=";")

    def test_required_columns_present(self, raw_df):
        expected = {
            "fixed acidity", "volatile acidity", "citric acid",
            "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates",
            "alcohol", "quality",
        }
        assert expected.issubset(set(raw_df.columns))

    def test_no_missing_values(self, raw_df):
        assert raw_df.isnull().sum().sum() == 0

    def test_quality_values_in_valid_range(self, raw_df):
        assert raw_df["quality"].between(0, 10).all()


class TestQualityBucketing:
    def test_bucket_quality_low(self):
        buckets = {"low": [0, 5], "medium": [6, 6], "high": [7, 10]}
        assert bucket_quality(3, buckets) == "low"
        assert bucket_quality(5, buckets) == "low"

    def test_bucket_quality_medium(self):
        buckets = {"low": [0, 5], "medium": [6, 6], "high": [7, 10]}
        assert bucket_quality(6, buckets) == "medium"

    def test_bucket_quality_high(self):
        buckets = {"low": [0, 5], "medium": [6, 6], "high": [7, 10]}
        assert bucket_quality(7, buckets) == "high"
        assert bucket_quality(10, buckets) == "high"

    def test_bucket_quality_raises_on_out_of_range(self):
        buckets = {"low": [0, 5], "medium": [6, 6], "high": [7, 10]}
        with pytest.raises(ValueError):
            bucket_quality(11, buckets)


class TestPreprocessing:
    def test_preprocess_adds_quality_class_column(self, raw_df, params):
        processed = preprocess(raw_df, params)
        assert "quality_class" in processed.columns

    def test_preprocess_produces_only_expected_labels(self, raw_df, params):
        processed = preprocess(raw_df, params)
        assert set(processed["quality_class"].unique()) <= {"low", "medium", "high"}

    def test_preprocess_row_count_unchanged(self, raw_df, params):
        processed = preprocess(raw_df, params)
        assert len(processed) == len(raw_df)

    def test_split_data_produces_correct_proportions(self, raw_df, params):
        processed = preprocess(raw_df, params)
        X_train, X_test, y_train, y_test = split_data(processed, params)

        total = len(X_train) + len(X_test)
        assert total == len(processed)

        expected_test_size = params["data"]["test_size"]
        actual_test_ratio = len(X_test) / total
        assert abs(actual_test_ratio - expected_test_size) < 0.02

    def test_split_data_excludes_target_columns_from_features(self, raw_df, params):
        processed = preprocess(raw_df, params)
        X_train, X_test, y_train, y_test = split_data(processed, params)

        assert "quality" not in X_train.columns
        assert "quality_class" not in X_train.columns