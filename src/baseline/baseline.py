"""XGBoost baseline: train, evaluate, log to MLflow, save plots and results.json."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.logger import get_logger

plt.switch_backend("Agg")

logger = get_logger(__name__)

FEATURE_COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def train_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    figures_dir: Path,
    results_path: Path,
    mlflow_experiment: str,
    xgb_params: dict[str, int | float | str | bool],
) -> XGBClassifier:
    """Train XGBoost, evaluate on test, log to MLflow, save metrics and figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values
    X_test = test[FEATURE_COLS].values
    y_test = test["target"].values

    mlflow.set_experiment(mlflow_experiment)
    with mlflow.start_run(run_name="baseline_xgb"):
        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba: np.ndarray = model.predict_proba(X_test)[:, 1]

        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1_weighted": float(
                f1_score(y_test, y_pred, average="weighted", zero_division=0)
            ),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
        mlflow.log_params({k: str(v) for k, v in xgb_params.items()})
        mlflow.log_metrics(metrics)
        logger.info("Baseline metrics: %s", metrics)

        _save_confusion_matrix(y_test, y_pred, figures_dir / "baseline_confusion.png")
        _save_roc_curve(model, X_test, y_test, figures_dir / "baseline_roc.png")
        _update_results_json(results_path, {"baseline_xgb": metrics})

    return model


def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path,
) -> None:
    """Save confusion matrix figure to *path*."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    ax.set_title("XGBoost Baseline — Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_roc_curve(
    model: XGBClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    path: Path,
) -> None:
    """Save ROC curve figure to *path*."""
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax)
    ax.set_title("XGBoost Baseline — ROC Curve")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _update_results_json(
    path: Path,
    new_data: dict[str, dict[str, float]],
) -> None:
    """Merge *new_data* into the existing results JSON, preserving other keys."""
    existing: dict[str, dict[str, float]] = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing.update(new_data)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)
