"""Tests for the preprocessing pipeline."""

from __future__ import annotations

import numpy as np

from churn import config
from churn.preprocess import build_preprocessor


def test_preprocessor_fits_and_transforms(training_frame):
    pre = build_preprocessor()
    out = pre.fit_transform(training_frame)
    # One row in, one row out; columns expand because of one-hot encoding.
    assert out.shape[0] == len(training_frame)
    assert out.shape[1] > len(config.NUMERIC_FEATURES)


def test_no_nans_after_transform(training_frame):
    # Inject a missing numeric value; the imputer must fill it.
    frame = training_frame.copy()
    frame.loc[frame.index[0], "TotalCharges"] = np.nan
    out = build_preprocessor().fit_transform(frame)
    assert not np.isnan(out).any()


def test_unknown_category_does_not_crash(training_frame):
    # handle_unknown="ignore" => a never-before-seen category is safely encoded.
    pre = build_preprocessor().fit(training_frame)
    novel = training_frame.iloc[[0]].copy()
    novel.loc[novel.index[0], "Contract"] = "Lifetime"  # unseen category
    out = pre.transform(novel)
    assert out.shape[0] == 1
