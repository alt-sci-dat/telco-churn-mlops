"""FastAPI service that serves churn predictions.

What is FastAPI? A framework for building web APIs in Python. An "API" is just a
way for other programs (a website, a mobile app, a cron job) to ask your model a
question over HTTP and get an answer back as JSON. FastAPI is popular because it
uses the Pydantic schemas to validate input automatically and generates
interactive docs for free (visit /docs when the server runs).

Design notes for production-readiness:
  * The model is loaded once at startup (loading on every request would be slow)
    via a dependency function `get_model`. Using FastAPI's dependency-injection
    means tests can swap in a tiny fake model with `app.dependency_overrides` —
    no real training run needed in CI.
  * Every incoming request's features are buffered in memory so the /drift
    endpoint can compare live traffic against the training distribution.
"""

from __future__ import annotations

from collections import deque
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException

from churn import __version__, config, drift
from churn.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CustomerFeatures,
    DriftResponse,
    HealthResponse,
    PredictionResponse,
)

app = FastAPI(
    title="Telco Churn Prediction API",
    description="Predict whether a customer will churn, with built-in drift monitoring.",
    version=__version__,
)

# Rolling buffer of the most recent requests' features, used for drift checks.
# A deque with maxlen automatically discards the oldest entries — bounded memory.
_REQUEST_BUFFER: deque[dict] = deque(maxlen=5000)


# --- Dependencies (loaded lazily + cached, overridable in tests) -------------
@lru_cache(maxsize=1)
def get_model():
    """Load the trained pipeline once and cache it."""
    if not config.MODEL_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {config.MODEL_PATH}. Run training first.",
        )
    return joblib.load(config.MODEL_PATH)


@lru_cache(maxsize=1)
def get_reference() -> dict:
    """Load the drift reference (training-data distribution) once and cache it."""
    if not config.REFERENCE_PATH.exists():
        return {}
    return drift.load_reference()


def _risk_level(probability: float) -> str:
    if probability >= 0.66:
        return "high"
    if probability >= 0.33:
        return "medium"
    return "low"


def _score(model, customers: list[CustomerFeatures]) -> list[PredictionResponse]:
    """Run the model on validated customers and shape the response."""
    # Pydantic objects -> DataFrame with exactly the columns the pipeline expects.
    frame = pd.DataFrame([c.model_dump() for c in customers])
    _REQUEST_BUFFER.extend(frame.to_dict(orient="records"))

    probabilities = model.predict_proba(frame[config.FEATURE_COLUMNS])[:, 1]
    results = []
    for prob in probabilities:
        prob = float(prob)
        results.append(
            PredictionResponse(
                churn_probability=round(prob, 4),
                churn_prediction=int(prob >= 0.5),
                risk_level=_risk_level(prob),
            )
        )
    return results


# --- Endpoints ---------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness/readiness probe. Deploy platforms ping this to check the app.

    `model_loaded` reflects whether the model actually deserializes, not just
    whether the file exists — a corrupt artifact should report unhealthy.
    """
    try:
        get_model()
        model_loaded = True
    except Exception:  # noqa: BLE001 - any load failure means "not ready"
        model_loaded = False
    return HealthResponse(status="ok", model_loaded=model_loaded, version=__version__)


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(customer: CustomerFeatures, model=Depends(get_model)) -> PredictionResponse:
    """Predict churn for a single customer."""
    return _score(model, [customer])[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["predictions"])
def predict_batch(
    request: BatchPredictionRequest, model=Depends(get_model)
) -> BatchPredictionResponse:
    """Predict churn for many customers in one call.

    Batch size is bounded by the schema (1..1000), so an empty or oversized
    request is rejected with a 422 before reaching here.
    """
    return BatchPredictionResponse(predictions=_score(model, request.customers))


@app.get("/drift", response_model=DriftResponse, tags=["monitoring"])
def drift_report(reference: dict = Depends(get_reference)) -> DriftResponse:
    """Compare recent live requests against the training-data distribution.

    Call /predict a bunch of times first to fill the buffer, then hit /drift to
    see whether incoming traffic has drifted away from what the model trained on.
    """
    if not reference:
        raise HTTPException(status_code=503, detail="Drift reference not available.")
    if not _REQUEST_BUFFER:
        return DriftResponse(
            drift_detected=False,
            n_samples=0,
            drifted_features=[],
            features={},
            message="No requests recorded yet. Call /predict first.",
        )

    incoming = pd.DataFrame(list(_REQUEST_BUFFER))
    report = drift.compute_drift(reference, incoming)
    msg = (
        f"Drift detected in: {', '.join(report['drifted_features'])}. Consider retraining."
        if report["drift_detected"]
        else "No significant drift detected."
    )
    return DriftResponse(
        drift_detected=report["drift_detected"],
        n_samples=report["n_samples"],
        drifted_features=report["drifted_features"],
        features=report["features"],
        message=msg,
    )


@app.get("/", tags=["ops"])
def root() -> dict:
    return {"message": "Telco Churn API. See /docs for interactive documentation."}
