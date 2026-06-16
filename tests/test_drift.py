"""Tests for drift detection. We verify PSI ~ 0 for identical data and high for shifted."""

from __future__ import annotations

from churn import drift


def test_no_drift_for_identical_data(training_frame):
    reference = drift.build_reference(training_frame)
    report = drift.compute_drift(reference, training_frame)
    # Same data in and out => no feature should flag as significantly drifted.
    assert report["drift_detected"] is False
    assert report["drifted_features"] == []


def test_drift_detected_for_shifted_numeric(training_frame):
    reference = drift.build_reference(training_frame)
    shifted = training_frame.copy()
    # Push tenure way out of the trained range -> distribution clearly changes.
    shifted["tenure"] = shifted["tenure"] + 500
    report = drift.compute_drift(reference, shifted)
    assert report["features"]["tenure"]["psi"] > drift.PSI_SIGNIFICANT
    assert "tenure" in report["drifted_features"]


def test_drift_detected_for_shifted_categorical(training_frame):
    reference = drift.build_reference(training_frame)
    shifted = training_frame.copy()
    # Make every customer fiber-optic -> categorical distribution collapses.
    shifted["InternetService"] = "Fiber optic"
    report = drift.compute_drift(reference, shifted)
    assert report["features"]["InternetService"]["psi"] > drift.PSI_MODERATE


def test_reference_roundtrip(training_frame, tmp_path):
    reference = drift.build_reference(training_frame)
    path = tmp_path / "ref.json"
    drift.save_reference(reference, path)
    loaded = drift.load_reference(path)
    assert loaded["numeric"].keys() == reference["numeric"].keys()
