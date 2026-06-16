"""Smoke test for the training workflow.

Training is the most complex module, so even a single-trial run through the whole
pipeline (data -> tune -> fit -> log -> save) is valuable: it catches regressions
like an MLflow API change or a broken artifact write without a network call or a
full tuning run. We monkeypatch the data loader (no download) and redirect every
artifact path into a tmp dir so nothing touches the committed model.
"""

from __future__ import annotations

import churn.train as train_mod
from churn import config


def test_train_smoke(monkeypatch, tmp_path, training_frame):
    # A small, already-clean, learnable dataset with the target column attached.
    df = training_frame.copy()
    df[config.TARGET] = (df["tenure"] < 12).astype(int)
    monkeypatch.setattr(train_mod.data, "load_dataset", lambda: df)

    # Redirect all artifact writes into tmp_path; chdir so MLflow's default
    # artifact folder (./mlartifacts) also lands in the tmp dir.
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(config, "REFERENCE_PATH", tmp_path / "reference.json")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)

    metrics = train_mod.train(n_trials=1)

    # The flow completed and produced sane outputs.
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert (tmp_path / "model.joblib").exists()
    assert (tmp_path / "reference.json").exists()
    assert (tmp_path / "metadata.json").exists()


def test_training_is_reproducible(monkeypatch, tmp_path, training_frame):
    """Seeded Optuna + seeded split/model => identical metrics across runs."""
    df = training_frame.copy()
    df[config.TARGET] = (df["tenure"] < 12).astype(int)
    monkeypatch.setattr(train_mod.data, "load_dataset", lambda: df)
    monkeypatch.setattr(config, "MODEL_DIR", tmp_path)
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(config, "METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(config, "REFERENCE_PATH", tmp_path / "reference.json")
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)

    first = train_mod.train(n_trials=3)
    second = train_mod.train(n_trials=3)
    assert first["roc_auc"] == second["roc_auc"]
