"""Pydantic models = the API's input/output contract.

"Input validation" means: before our model ever sees a request, we check that it
has the right fields, the right types, and sensible values. If a caller sends
`tenure: -5` or `Contract: "Forever"`, FastAPI rejects it automatically with a
clear 422 error — the bad data never reaches the model. This is both a safety
feature (no garbage predictions) and self-documenting (the schema IS the API doc).

We use `Literal[...]` for categorical fields so only the exact categories seen in
training are accepted, and numeric `Field(...)` constraints to bound the numbers.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    """One customer's attributes — exactly what the model needs to score them."""

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=120, description="Months with the company")
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(..., ge=0, le=1000)
    TotalCharges: float = Field(..., ge=0, le=100000)

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., ge=0, le=1)
    churn_prediction: int = Field(..., description="1 = likely to churn, 0 = likely to stay")
    risk_level: Literal["low", "medium", "high"]


class BatchPredictionRequest(BaseModel):
    # Bound the batch size at the validation boundary: an unbounded list lets one
    # request exhaust memory on a small instance (DoS). min_length=1 also makes
    # an empty batch a clean 422 instead of needing a manual check downstream.
    customers: list[CustomerFeatures] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class DriftResponse(BaseModel):
    drift_detected: bool
    n_samples: int
    drifted_features: list[str]
    features: dict
    message: str
