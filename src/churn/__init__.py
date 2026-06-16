"""Telco customer churn prediction package.

Modules:
    config      paths, feature lists, constants (single source of truth)
    data        download + clean the raw dataset
    preprocess  build the scikit-learn preprocessing pipeline
    train       train XGBoost with Optuna tuning, log to MLflow, save artifacts
    drift       detect data drift between training data and live requests
    schemas     Pydantic request/response models used by the API
"""

__version__ = "0.1.0"
