"""Shared pytest fixtures.

`conftest.py` is auto-discovered by pytest; anything defined here is available to
every test file without importing it. We build small, fast, synthetic data so the
whole suite runs in seconds and needs NO dataset download or real training run —
that's what lets CI stay fast and free.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from churn import config
from churn.preprocess import build_preprocessor


def _make_raw_row(churn: str, total_charges: str) -> dict:
    """A single raw-format Telco row (strings included, like the real CSV)."""
    return {
        "customerID": "0001-AAAA",
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
        "TotalCharges": total_charges,  # may be a blank string, like the real data
        "Churn": churn,
    }


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Tiny raw dataframe that includes the blank-TotalCharges quirk."""
    rows = [
        _make_raw_row("Yes", "445.50"),
        _make_raw_row("No", "1234.00"),
        _make_raw_row("No", " "),  # the known blank-string quirk (tenure-0 cust)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def training_frame() -> pd.DataFrame:
    """A larger synthetic, already-clean feature frame for fitting models/drift."""
    rng = np.random.default_rng(0)
    n = 200
    data = {
        "tenure": rng.integers(0, 72, n),
        "MonthlyCharges": rng.uniform(20, 120, n),
        "TotalCharges": rng.uniform(20, 8000, n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "gender": rng.choice(["Female", "Male"], n),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n),
        "PhoneService": rng.choice(["Yes", "No"], n),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            n,
        ),
    }
    return pd.DataFrame(data)[config.FEATURE_COLUMNS]


@pytest.fixture
def fitted_pipeline(training_frame) -> Pipeline:
    """A real (but tiny/fast) trained pipeline for API tests."""
    # Make a learnable target: short-tenure month-to-month customers churn more.
    y = (
        (training_frame["tenure"] < 12)
        & (training_frame["Contract"] == "Month-to-month")
    ).astype(int)
    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor()),
            ("model", XGBClassifier(n_estimators=20, max_depth=3, eval_metric="logloss")),
        ]
    )
    pipeline.fit(training_frame, y)
    return pipeline
