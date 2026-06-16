"""Build the scikit-learn preprocessing pipeline.

A "pipeline" chains together the steps that turn raw columns into the numeric
matrix a model needs. The huge benefit: the *exact same* transformations applied
during training are applied automatically at prediction time. No "I forgot to
encode the new data the same way" bugs — the #1 cause of broken ML in production.

We use a ColumnTransformer, which applies different steps to different columns:
  - Numeric columns: fill missing values with the median.
  - Categorical columns: one-hot encode (turn "Contract=Two year" into a 0/1
    column the model can read). `handle_unknown="ignore"` means if a live request
    contains a category we never saw in training, we don't crash — we encode it
    as all-zeros.

Note: we deliberately do NOT scale numeric features. XGBoost is a tree model; it
splits on thresholds, so the scale of a feature doesn't matter. Scaling would
add complexity for zero benefit here.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from . import config


def build_preprocessor() -> ColumnTransformer:
    """Return an unfitted ColumnTransformer for the Telco features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            # Some categoricals could in theory be missing too; be safe.
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, config.NUMERIC_FEATURES),
            ("cat", categorical_pipeline, config.CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # ignore any column not explicitly listed
    )
    return preprocessor
