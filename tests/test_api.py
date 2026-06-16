"""Tests for the FastAPI service.

We use FastAPI's TestClient (no real server needed) and override the model
dependency with the tiny `fitted_pipeline` fixture, so these tests run fast and
don't require a trained model file on disk.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import _REQUEST_BUFFER, app, get_model, get_reference
from churn import drift

VALID_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.10,
    "TotalCharges": 445.50,
}


@pytest.fixture
def client(fitted_pipeline, training_frame):
    """A TestClient wired to the fake model + a real drift reference."""
    app.dependency_overrides[get_model] = lambda: fitted_pipeline
    app.dependency_overrides[get_reference] = lambda: drift.build_reference(training_frame)
    _REQUEST_BUFFER.clear()  # isolate the in-memory buffer between tests
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_valid(client):
    resp = client.post("/predict", json=VALID_CUSTOMER)
    assert resp.status_code == 200
    body = resp.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_prediction"] in (0, 1)
    assert body["risk_level"] in ("low", "medium", "high")


def test_predict_rejects_bad_category(client):
    bad = {**VALID_CUSTOMER, "Contract": "Forever"}  # not a valid enum value
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422  # validation error, model never sees it


def test_predict_rejects_negative_tenure(client):
    bad = {**VALID_CUSTOMER, "tenure": -3}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_rejects_missing_field(client):
    bad = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_batch_predict(client):
    resp = client.post("/predict/batch", json={"customers": [VALID_CUSTOMER, VALID_CUSTOMER]})
    assert resp.status_code == 200
    assert len(resp.json()["predictions"]) == 2


def test_batch_rejects_empty(client):
    # min_length=1 => an empty batch is a clean validation error.
    resp = client.post("/predict/batch", json={"customers": []})
    assert resp.status_code == 422


def test_batch_rejects_oversized(client):
    # max_length=1000 guards against a memory-exhaustion request.
    resp = client.post("/predict/batch", json={"customers": [VALID_CUSTOMER] * 1001})
    assert resp.status_code == 422


def test_drift_endpoint_after_requests(client):
    # No requests yet -> graceful empty report.
    assert client.get("/drift").json()["n_samples"] == 0
    # Make some predictions, then drift should have samples to analyse.
    for _ in range(5):
        client.post("/predict", json=VALID_CUSTOMER)
    report = client.get("/drift").json()
    assert report["n_samples"] >= 5
    assert "features" in report


def test_metadata_returns_metrics(client):
    resp = client.get("/metadata")
    assert resp.status_code == 200
    body = resp.json()
    assert "metrics" in body
    assert "roc_auc" in body["metrics"]
