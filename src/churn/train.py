"""Train the churn model: XGBoost + Optuna tuning + MLflow tracking.

Plain-English tour of the three tools:

- XGBoost: a "gradient boosted trees" model. It builds many small decision trees
  one after another, each one correcting the mistakes of the previous ones. It's
  the go-to model for tabular data (rows and columns like a spreadsheet) and wins
  most Kaggle competitions on this kind of data.

- Optuna: hyperparameter tuning. A model has "knobs" (how deep each tree is, how
  fast it learns, etc.) that you set *before* training — these are
  hyperparameters. Optuna automatically tries many combinations and keeps the
  best, instead of you guessing by hand. Each attempt is called a "trial".

- MLflow: experiment tracking. Every time you train, you produce metrics, params,
  and a model file. MLflow records all of it so you can answer "which run was
  best and exactly how was it configured?" weeks later. It's your lab notebook.

The output is ONE artifact: a scikit-learn Pipeline = preprocessing + model
glued together. The API loads that single file and the raw-data-in,
prediction-out logic just works.
"""

from __future__ import annotations

import json

import joblib
import mlflow
import optuna
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config, data, drift
from .preprocess import build_preprocessor


def _build_pipeline(params: dict, scale_pos_weight: float) -> Pipeline:
    """Glue preprocessing + an XGBoost classifier into one Pipeline."""
    model = XGBClassifier(
        **params,
        scale_pos_weight=scale_pos_weight,  # see note in train() about imbalance
        eval_metric="logloss",
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocess", build_preprocessor()), ("model", model)])


def _objective(trial: optuna.Trial, X, y, scale_pos_weight: float) -> float:
    """One Optuna trial: pick hyperparameters, score them with cross-validation.

    Cross-validation (CV) splits the data into k folds, trains on k-1 and tests
    on the held-out fold, k times. Averaging the scores gives a robust estimate
    that doesn't depend on one lucky train/test split. We optimise ROC-AUC, which
    measures how well the model ranks churners above non-churners (good for
    imbalanced data where plain accuracy is misleading).
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
    }
    pipeline = _build_pipeline(params, scale_pos_weight)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean())


def train(n_trials: int = 30, experiment_name: str = "telco-churn") -> dict:
    """Run the full training workflow and persist all artifacts.

    Returns a dict of test-set metrics.
    """
    # --- 1. Load + clean data, then split into train/test ---
    df = data.load_dataset()
    X, y = data.split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,  # keep the same churn ratio in both splits
    )

    # Class imbalance: only ~27% of customers churn. `scale_pos_weight` tells
    # XGBoost to pay proportionally more attention to the rarer "churn" class.
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    scale_pos_weight = n_neg / max(n_pos, 1)

    # --- 2. Point MLflow at a local SQLite DB (no server, completely free) ---
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    # --- 3. Tune hyperparameters with Optuna ---
    # `direction="maximize"` because higher ROC-AUC is better. We seed the sampler
    # so the search is deterministic: combined with the seeded split and model,
    # `make train` produces the exact same model every run (the reproducibility
    # promised in config.RANDOM_STATE).
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE),
    )
    study.optimize(
        lambda t: _objective(t, X_train, y_train, scale_pos_weight),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    best_params = study.best_params

    # --- 4. Train the final model on ALL training data with the best params,
    #        and log everything to MLflow ---
    with mlflow.start_run(run_name="best-model"):
        pipeline = _build_pipeline(best_params, scale_pos_weight)
        pipeline.fit(X_train, y_train)

        # Evaluate on the untouched test set — the honest estimate of real-world
        # performance, since the model never saw this data during training/tuning.
        proba = pipeline.predict_proba(X_test)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": roc_auc_score(y_test, proba),
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "cv_roc_auc": study.best_value,
        }

        mlflow.log_params(best_params)
        mlflow.log_param("scale_pos_weight", round(scale_pos_weight, 4))
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.sklearn.log_model(pipeline, name="model")

        # --- 5. Persist artifacts the API will load ---
        config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, config.MODEL_PATH)

        # Build + save the drift reference from the TRAINING features. Live data
        # gets compared against this baseline.
        reference = drift.build_reference(X_train)
        drift.save_reference(reference)

        # Save human-readable metadata next to the model.
        config.METADATA_PATH.write_text(
            json.dumps(
                {
                    "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
                    "best_params": best_params,
                    "n_train": int(len(X_train)),
                    "n_test": int(len(X_test)),
                    "churn_rate": round(float(y.mean()), 4),
                },
                indent=2,
            )
        )

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the churn model.")
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Number of Optuna tuning trials (more = better but slower).",
    )
    args = parser.parse_args()

    print(f"Training with {args.n_trials} Optuna trials...")
    results = train(n_trials=args.n_trials)
    print("\nTest-set performance:")
    for name, value in results.items():
        print(f"  {name:12s}: {value:.4f}")
    print(f"\nModel saved to: {config.MODEL_PATH}")
    print(f"Drift reference saved to: {config.REFERENCE_PATH}")
    print("View experiments with: mlflow ui --backend-store-uri sqlite:///mlflow.db")
