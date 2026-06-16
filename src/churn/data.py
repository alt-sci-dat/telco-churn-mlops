"""Download and clean the Telco Customer Churn dataset.

A "data pipeline" just means: get raw data -> make it consistent and correct ->
hand clean data to the model. Garbage in, garbage out, so this step matters as
much as the model itself.

The raw Telco dataset has two well-known quirks we fix here:
  1. `TotalCharges` is stored as text and has 11 blank values (customers who
     just joined, tenure = 0). We coerce it to a number and impute later.
  2. The target `Churn` is "Yes"/"No" text; models need numbers (1/0).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def download_data(force: bool = False) -> pd.DataFrame:
    """Ensure the raw CSV exists locally, then return it as a DataFrame.

    Resolution order (first that works wins):
      1. A file already on disk at config.RAW_CSV (cached from a prior run).
      2. Public GitHub mirrors of the dataset (no account needed).
      3. The Kaggle API, if you have ~/.kaggle/kaggle.json configured.
    """
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)

    if config.RAW_CSV.exists() and not force:
        return pd.read_csv(config.RAW_CSV)

    # --- Try public mirrors first (no credentials required) ---
    for url in config.DATASET_URLS:
        try:
            df = pd.read_csv(url)
            df.to_csv(config.RAW_CSV, index=False)
            return df
        except Exception as exc:  # noqa: BLE001 - try the next source on any failure
            logger.warning("Dataset mirror failed (%s): %s", url, exc)
            continue

    # --- Fall back to Kaggle if the user has it set up ---
    try:
        import kagglehub  # type: ignore

        path = kagglehub.dataset_download(config.KAGGLE_DATASET)
        # kagglehub returns a folder; find the CSV inside it.
        csvs = list(Path(path).glob("*.csv"))
        if csvs:
            df = pd.read_csv(csvs[0])
            df.to_csv(config.RAW_CSV, index=False)
            return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("Kaggle fallback failed: %s", exc)

    raise RuntimeError(
        "Could not download the dataset from any public mirror or Kaggle. "
        "Manually download 'WA_Fn-UseC_-Telco-Customer-Churn.csv' from Kaggle "
        f"and save it to: {config.RAW_CSV}"
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the fixes the raw data needs. Pure function: no side effects.

    Returns a cleaned copy with a numeric `Churn` column (1 = churned).
    Keeping this as a pure function (input df -> output df, no file reads) makes
    it trivial to unit-test with a tiny hand-made DataFrame.
    """
    df = df.copy()

    # Quirk #1: TotalCharges is text with blank strings -> coerce to float.
    # `errors="coerce"` turns un-parseable values (the blanks) into NaN, which
    # the preprocessing pipeline will impute. We do NOT drop these rows: a
    # production model must handle brand-new customers (tenure 0) gracefully.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Quirk #2: the target is "Yes"/"No" -> map to 1/0.
    if config.TARGET in df.columns:
        df[config.TARGET] = (
            df[config.TARGET].map({"Yes": 1, "No": 0}).astype("int64")
        )

    return df


def load_dataset() -> pd.DataFrame:
    """Convenience: download (or load cached) + clean in one call."""
    return clean_data(download_data())


def split_features_target(df: pd.DataFrame):
    """Separate the model inputs (X) from what we predict (y).

    We keep only the columns the model is allowed to use (FEATURE_COLUMNS),
    which automatically drops customerID and anything unexpected.
    """
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET].copy()
    return X, y
