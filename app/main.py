"""
FastAPI application serving the trained Wine Quality classifier.

Endpoints:
    GET  /         - basic application info
    GET  /health   - health check (verifies model is loaded)
    POST /predict  - predict quality class from wine features
    GET  /metrics  - Prometheus-compatible metrics

Run locally:
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""

import time
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

MODEL_PATH = Path("models/model.pkl")
MODEL_VERSION = "1.0.0"

app = FastAPI(
    title="Wine Quality Classifier API",
    description="Serves a Random Forest model predicting wine quality class (low/medium/high).",
    version=MODEL_VERSION,
)

# --- Load model once at startup ---
model = None
model_load_error: Optional[str] = None
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model_load_error = str(e)

# --- Simple in-memory metrics (exposed via /metrics) ---
metrics_state = {
    "request_count": 0,
    "prediction_count": 0,
    "error_count": 0,
    "total_latency_seconds": 0.0,
    "status_code_counts": {},
}


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(..., alias="fixed acidity")
    volatile_acidity: float = Field(..., alias="volatile acidity")
    citric_acid: float = Field(..., alias="citric acid")
    residual_sugar: float = Field(..., alias="residual sugar")
    chlorides: float = Field(...)
    free_sulfur_dioxide: float = Field(..., alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(..., alias="total sulfur dioxide")
    density: float = Field(...)
    pH: float = Field(...)
    sulphates: float = Field(...)
    alcohol: float = Field(...)

    model_config = {"populate_by_name": True}


class PredictionResponse(BaseModel):
    prediction: str
    model_version: str


@app.middleware("http")
async def track_metrics(request, call_next):
    start = time.time()
    metrics_state["request_count"] += 1
    try:
        response = await call_next(request)
    except Exception:
        metrics_state["error_count"] += 1
        raise
    elapsed = time.time() - start
    metrics_state["total_latency_seconds"] += elapsed
    status = str(response.status_code)
    metrics_state["status_code_counts"][status] = (
        metrics_state["status_code_counts"].get(status, 0) + 1
    )
    if response.status_code >= 400:
        metrics_state["error_count"] += 1
    return response


@app.get("/")
def root():
    return {
        "app": "Wine Quality Classifier API",
        "version": MODEL_VERSION,
        "model_loaded": model is not None,
        "endpoints": ["/", "/health", "/predict", "/metrics"],
    }


@app.get("/health")
def health():
    if model is None:
        return Response(
            content='{"status": "unhealthy", "reason": "model not loaded"}',
            status_code=503,
            media_type="application/json",
        )
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: WineFeatures):
    if model is None:
        return Response(
            content='{"detail": "Model is not available for predictions"}',
            status_code=503,
            media_type="application/json",
        )

    feature_order = [
        "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
        "chlorides", "free sulfur dioxide", "total sulfur dioxide",
        "density", "pH", "sulphates", "alcohol",
    ]
    data = features.model_dump(by_alias=True)
    row = [[data[col] for col in feature_order]]

    prediction = model.predict(row)[0]
    metrics_state["prediction_count"] += 1

    return PredictionResponse(prediction=prediction, model_version=MODEL_VERSION)


@app.get("/metrics")
def metrics():
    m = metrics_state
    avg_latency = (
        m["total_latency_seconds"] / m["request_count"] if m["request_count"] > 0 else 0.0
    )

    lines = [
        "# HELP app_request_count_total Total number of requests received",
        "# TYPE app_request_count_total counter",
        f'app_request_count_total {m["request_count"]}',
        "# HELP app_prediction_count_total Total number of predictions made",
        "# TYPE app_prediction_count_total counter",
        f'app_prediction_count_total {m["prediction_count"]}',
        "# HELP app_error_count_total Total number of error responses (status >= 400)",
        "# TYPE app_error_count_total counter",
        f'app_error_count_total {m["error_count"]}',
        "# HELP app_request_latency_seconds_avg Average request latency in seconds",
        "# TYPE app_request_latency_seconds_avg gauge",
        f"app_request_latency_seconds_avg {avg_latency:.6f}",
    ]
    for status, count in m["status_code_counts"].items():
        lines.append(f'app_status_code_count{{code="{status}"}} {count}')

    return Response(content="\n".join(lines) + "\n", media_type="text/plain")