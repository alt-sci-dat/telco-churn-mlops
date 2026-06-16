"""Tests for data cleaning. These lock in the fixes for the dataset's quirks."""

from __future__ import annotations

import pandas as pd

from churn import config
from churn.data import clean_data, split_features_target


def test_total_charges_coerced_to_numeric(raw_df):
    cleaned = clean_data(raw_df)
    assert pd.api.types.is_numeric_dtype(cleaned["TotalCharges"])


def test_blank_total_charges_becomes_nan(raw_df):
    # The third row had a blank " " — it must become NaN, not crash.
    cleaned = clean_data(raw_df)
    assert cleaned["TotalCharges"].isna().sum() == 1


def test_target_mapped_to_int(raw_df):
    cleaned = clean_data(raw_df)
    assert set(cleaned[config.TARGET].unique()) <= {0, 1}
    assert pd.api.types.is_integer_dtype(cleaned[config.TARGET])


def test_clean_data_is_pure(raw_df):
    # Cleaning must not mutate the caller's dataframe.
    before = raw_df["TotalCharges"].copy()
    clean_data(raw_df)
    pd.testing.assert_series_equal(raw_df["TotalCharges"], before)


def test_split_drops_id_and_target(raw_df):
    X, y = split_features_target(clean_data(raw_df))
    assert config.ID_COL not in X.columns
    assert config.TARGET not in X.columns
    assert list(X.columns) == config.FEATURE_COLUMNS
    assert len(X) == len(y)
