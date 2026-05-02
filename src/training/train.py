"""Train ConformalXGBoost, calibrate on the cal split, log to MLflow.

CLI: ``python -m src.training.train --config config/config.yaml``

This is the OPUS-grade Day 3 training entrypoint. All design rules from
CLAUDE.md that apply: C35 (calibrate on cal only), C36 (safe_predict guards
inside the model), C37 (assert empirical coverage >= 1 - alpha - tolerance),
C23 (training_stats reloadable for skew checks).
"""

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml  # type: ignore[import-untyped]

from src.data.preprocessing import FEATURE_COLS
from src.data.skew_check import save_training_stats
from src.logger import get_logger
from src.models.model import ConformalXGBoost

logger = get_logger(__name__)

COVERAGE_TOLERANCE = 0.05  # rule C37 — finite-sample slack


def run_training(cfg: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Train + calibrate + log; return per-alpha {coverage, mean_set_size}.

    Reads ``data/processed/{train,cal}.csv`` (already StandardScaler-transformed
    by Day 2's prepare_data.py — we do NOT re-scale here). Writes the model,
    training stats, and calibration metadata to ``cfg['paths']['models_dir']``.
    """
    processed_dir = Path(cfg["data"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    alphas: list[float] = list(cfg["training"]["alphas"])
    seed: int = int(cfg["data"]["random_seed"])

    train = pd.read_csv(processed_dir / "train.csv")
    cal = pd.read_csv(processed_dir / "cal.csv")

    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values.astype(int)
    X_cal = cal[FEATURE_COLS].values
    y_cal = cal["target"].values.astype(int)

    xgb_cfg: dict[str, Any] = dict(cfg["model"]["xgb"])
    # use_label_encoder was removed in XGBoost 2.0+; drop it if present.
    xgb_cfg.pop("use_label_encoder", None)

    model = ConformalXGBoost(
        n_estimators=int(xgb_cfg["n_estimators"]),
        max_depth=int(xgb_cfg["max_depth"]),
        learning_rate=float(xgb_cfg["learning_rate"]),
        seed=seed,
    )
    model.fit(X_train, y_train).calibrate(X_cal, y_cal, alphas=alphas)

    models_dir.mkdir(parents=True, exist_ok=True)
    save_training_stats(
        X_train, models_dir / "training_stats.json", feature_names=FEATURE_COLS
    )

    metadata: dict[str, Any] = {
        "split_used": "cal",
        "conformity_score": "lac",
        "prefit": True,
        "n_calibration_samples": int(len(cal)),
        "alphas": alphas,
    }
    with open(models_dir / "calibration_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Wrote calibration_metadata.json -> %s", models_dir)

    coverage_results: dict[str, dict[str, float]] = {}
    mlflow.set_experiment(cfg["training"]["mlflow_experiment"])
    with mlflow.start_run(run_name="conformal_xgb_lac"):
        mlflow.log_params({k: str(v) for k, v in xgb_cfg.items()})
        mlflow.log_params(
            {
                "conformity_score": "lac",
                "prefit": True,
                "seed": seed,
                "n_train": len(train),
                "n_cal": len(cal),
            }
        )
        for alpha in alphas:
            _, y_set = model.safe_predict(X_cal, alpha=alpha)
            covered = float(y_set[np.arange(len(y_cal)), y_cal].sum() / len(y_cal))
            mean_size = float(y_set.sum(axis=1).mean())
            mlflow.log_metric(f"empirical_coverage_alpha_{alpha}", covered)
            mlflow.log_metric(f"mean_set_size_alpha_{alpha}", mean_size)
            assert covered >= 1 - alpha - COVERAGE_TOLERANCE, (
                f"Coverage {covered:.3f} below 1-alpha-tol="
                f"{1 - alpha - COVERAGE_TOLERANCE:.3f} for alpha={alpha} (rule C37)"
            )
            coverage_results[str(alpha)] = {
                "empirical_coverage": covered,
                "mean_set_size": mean_size,
            }
            logger.info(
                "alpha=%.2f coverage=%.3f mean_set_size=%.2f",
                alpha,
                covered,
                mean_size,
            )

    model.save(models_dir)
    logger.info("Model + meta saved to %s", models_dir)
    return coverage_results


def main(config_path: str) -> None:
    """Load config and dispatch to run_training."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_training(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
