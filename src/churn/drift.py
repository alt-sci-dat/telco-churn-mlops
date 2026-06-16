"""Data drift monitoring.

What is data drift? Your model learned patterns from training data. If the data
flowing into the live API starts to look *different* from that training data
(say, suddenly everyone signing up is on fiber internet, or charges jump), the
model's predictions get unreliable — even though no code changed and no error is
thrown. Drift monitoring is your smoke detector for "the world changed and my
model is now quietly wrong."

We measure drift with the Population Stability Index (PSI), an industry-standard
metric (common in credit scoring). Intuition: bucket a feature into bins, compare
the % of data in each bin for training ("expected") vs live ("actual"). If the
distributions match, PSI ~ 0. The bigger the shift, the bigger the PSI.

Rule of thumb:
    PSI < 0.1   -> no significant drift
    0.1 - 0.25  -> moderate drift, keep an eye on it
    PSI > 0.25  -> significant drift, consider retraining

For categorical features we treat each category as its own bin.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Drift thresholds (see module docstring).
PSI_MODERATE = 0.10
PSI_SIGNIFICANT = 0.25
# Tiny constant so we never take log(0) when a bin is empty.
_EPS = 1e-6


def build_reference(df: pd.DataFrame, n_bins: int = 10) -> dict:
    """Summarise the *training* data so we can compare live data against it later.

    For numeric features we store quantile bin edges + the proportion of data in
    each bin. For categorical features we store the proportion of each category.
    This summary is small (a few KB of JSON) and ships alongside the model.
    """
    reference: dict = {"numeric": {}, "categorical": {}, "n_bins": n_bins}

    for col in config.NUMERIC_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        # Quantile-based edges adapt to the data's actual shape.
        edges = np.unique(np.quantile(series, np.linspace(0, 1, n_bins + 1)))
        # A constant (or near-constant) feature collapses to <2 unique edges, which
        # np.histogram can't bin. Fall back to a single bin spanning the value.
        if len(edges) < 2:
            edges = np.array([series.min(), series.min() + 1e-9])
        counts, _ = np.histogram(series, bins=edges)
        proportions = counts / max(counts.sum(), 1)
        reference["numeric"][col] = {
            "edges": edges.tolist(),
            "proportions": proportions.tolist(),
        }

    for col in config.CATEGORICAL_FEATURES:
        proportions = df[col].astype(str).value_counts(normalize=True)
        reference["categorical"][col] = proportions.to_dict()

    return reference


def _psi(expected: np.ndarray, actual: np.ndarray) -> float:
    """Population Stability Index between two proportion vectors."""
    expected = np.clip(expected, _EPS, None)
    actual = np.clip(actual, _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_psi(ref: dict, values: pd.Series) -> float:
    edges = np.array(ref["edges"])
    expected = np.array(ref["proportions"])
    values = pd.to_numeric(values, errors="coerce").dropna()
    # Clamp into the trained range so out-of-range values land in the edge bins
    # instead of being silently dropped by np.histogram. Otherwise live data that
    # shifts *beyond* the training max would understate drift — the opposite of
    # what we want from a drift monitor.
    clamped = np.clip(values.to_numpy(), edges[0], edges[-1])
    counts, _ = np.histogram(clamped, bins=edges)
    actual = counts / max(counts.sum(), 1)
    return _psi(expected, actual)


def _categorical_psi(ref: dict, values: pd.Series) -> float:
    actual_props = values.astype(str).value_counts(normalize=True).to_dict()
    categories = set(ref) | set(actual_props)
    expected = np.array([ref.get(c, 0.0) for c in categories])
    actual = np.array([actual_props.get(c, 0.0) for c in categories])
    return _psi(expected, actual)


def _label(psi: float) -> str:
    if psi >= PSI_SIGNIFICANT:
        return "significant"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "none"


def compute_drift(reference: dict, incoming: pd.DataFrame) -> dict:
    """Compare a batch of live data against the training reference.

    Returns a per-feature report plus an overall flag. `incoming` should be a
    DataFrame of raw feature rows (the same columns the model consumes).
    """
    report: dict = {"features": {}, "n_samples": int(len(incoming))}
    drifted = []

    for col, ref in reference.get("numeric", {}).items():
        if col not in incoming:
            continue
        psi = _numeric_psi(ref, incoming[col])
        report["features"][col] = {"psi": round(psi, 4), "drift": _label(psi)}
        if psi >= PSI_SIGNIFICANT:
            drifted.append(col)

    for col, ref in reference.get("categorical", {}).items():
        if col not in incoming:
            continue
        psi = _categorical_psi(ref, incoming[col])
        report["features"][col] = {"psi": round(psi, 4), "drift": _label(psi)}
        if psi >= PSI_SIGNIFICANT:
            drifted.append(col)

    report["drifted_features"] = drifted
    report["drift_detected"] = len(drifted) > 0
    return report


def save_reference(reference: dict, path: Path | None = None) -> None:
    path = path or config.REFERENCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(reference, indent=2))


def load_reference(path: Path | None = None) -> dict:
    path = path or config.REFERENCE_PATH
    return json.loads(Path(path).read_text())
