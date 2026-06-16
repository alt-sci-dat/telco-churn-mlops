"""Single source of truth for paths, column names, and constants.

Why a config module? In a real project, the data scientist, the API, the tests,
and the training script all need to agree on things like "which columns are
numeric" and "where is the model saved". Putting those in one place means you
change them once, not in five files.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. Everything is anchored to the project root so the code works no matter
# what directory you run it from. `Path(__file__)` is this file; we go up to the
# repo root (src/churn/config.py -> src/churn -> src -> repo root).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
# MLflow tracking: a local SQLite database (free, no server). Recent MLflow
# versions deprecated the old plain-folder store, so we use sqlite, which also
# unlocks the model registry. Artifacts (the saved models) go to ./mlartifacts.
MLFLOW_DB = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB}")

RAW_CSV = RAW_DIR / "telco_churn.csv"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
REFERENCE_PATH = MODEL_DIR / "drift_reference.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Allow overriding the model location via env var (used by Docker / deployment).
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))
REFERENCE_PATH = Path(os.getenv("REFERENCE_PATH", str(REFERENCE_PATH)))

# ---------------------------------------------------------------------------
# Dataset details. The IBM "Telco Customer Churn" dataset (same one Kaggle hosts).
# We list public mirrors so the download works without a Kaggle account.
# ---------------------------------------------------------------------------
DATASET_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]
# If you prefer Kaggle: set up ~/.kaggle/kaggle.json and the loader will use it.
KAGGLE_DATASET = "blastchar/telco-customer-churn"

# ---------------------------------------------------------------------------
# Schema. Which columns mean what. The model never sees customerID.
# ---------------------------------------------------------------------------
ID_COL = "customerID"
TARGET = "Churn"

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Reproducibility: a fixed random seed means you get the same train/test split
# and the same model every time you run training. Essential for debugging.
RANDOM_STATE = 42
TEST_SIZE = 0.2
