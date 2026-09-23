"""
Tests for the FastAPI application endpoints.

Uses FastAPI's TestClient, so no live server is required -
this is what runs inside GitHub Actions CI.

Run:
    pytest tests/test_api.py -v
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_WINE_SAMPLE = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4,
}


class TestRootEndpoint:
    def test_root_returns_200(self):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_returns_expected_fields(self):
        response = client.get("/")
        data = response.json()
        assert "app" in data
        assert "version" in data
        assert "model_loaded" in data
        assert "endpoints" in data


class TestHealthEndpoint:
    def test_health_returns_200_when_model_loaded(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_healthy_status(self):
        response = client.get("/health")
        assert response.json() == {"status": "healthy"}


class TestPredictEndpoint:
    def test_predict_with_valid_input_returns_200(self):
        response = client.post("/predict", json=VALID_WINE_SAMPLE)
        assert response.status_code == 200

    def test_predict_returns_expected_fields(self):
        response = client.post("/predict", json=VALID_WINE_SAMPLE)
        data = response.json()
        assert "prediction" in data
        assert "model_version" in data

    def test_predict_returns_valid_class_label(self):
        response = client.post("/predict", json=VALID_WINE_SAMPLE)
        data = response.json()
        assert data["prediction"] in {"low", "medium", "high"}

    def test_predict_known_sample_matches_expected_class(self):
        # This exact row is from the training data with quality=5,
        # which buckets to "low" - a known ground-truth check.
        response = client.post("/predict", json=VALID_WINE_SAMPLE)
        assert response.json()["prediction"] == "low"

    def test_predict_with_missing_fields_returns_422(self):
        incomplete = {"fixed acidity": 7.4}
        response = client.post("/predict", json=incomplete)
        assert response.status_code == 422

    def test_predict_with_wrong_type_returns_422(self):
        bad_sample = dict(VALID_WINE_SAMPLE)
        bad_sample["alcohol"] = "not-a-number"
        response = client.post("/predict", json=bad_sample)
        assert response.status_code == 422

    def test_predict_with_empty_body_returns_422(self):
        response = client.post("/predict", json={})
        assert response.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_returns_200(self):
        response = client.get("/metrics")
        assert response.status_code == 200

    def test_metrics_contains_expected_metric_names(self):
        response = client.get("/metrics")
        body = response.text
        assert "app_request_count_total" in body
        assert "app_prediction_count_total" in body
        assert "app_error_count_total" in body
        assert "app_request_latency_seconds_avg" in body

    def test_metrics_is_plain_text(self):
        response = client.get("/metrics")
        assert "text/plain" in response.headers["content-type"]