"""CLI orchestrator for Day 4 evaluation suite.

Runs the per-alpha coverage analysis on the test split, persists results to
``reports/results.json`` (merging with the Day 2 ``baseline_xgb`` block), and
logs metrics plus generated figures to MLflow as a ``full_evaluation`` run.

Subsequent PRs will extend this orchestrator with method-comparison,
Mondrian, ECE, DCA, and selective-classification sections; the early
versions therefore only know about the modules that have already landed.

CLI: ``python -m src.evaluation.evaluate --config config/config.yaml``.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import yaml  # type: ignore[import-untyped]

from src.data.preprocessing import FEATURE_COLS
from src.evaluation.coverage import (
    compute_per_alpha_coverage,
    plot_coverage_bar,
    plot_set_sizes_hist,
)
from src.evaluation.method_compare import compare_methods, plot_method_comparison
from src.logger import get_logger
from src.models.model import ConformalXGBoost

plt.switch_backend("Agg")
logger = get_logger(__name__)


def run_evaluation(cfg: dict[str, Any]) -> dict[str, Any]:
    """Execute the evaluation pipeline and return the merged results dict.

    Reads the already-scaled test split (Day 2's prepare_data.py wrote
    ``data/processed/{train,cal,test}.csv`` post-scaling, so we do NOT
    re-apply the StandardScaler here). Loads the calibrated model from
    ``cfg['paths']['models_dir']``.
    """
    processed_dir = Path(cfg["data"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    reports_dir = Path(cfg["paths"]["reports_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    alphas: list[float] = list(cfg["training"]["alphas"])

    train = pd.read_csv(processed_dir / "train.csv")
    cal = pd.read_csv(processed_dir / "cal.csv")
    test = pd.read_csv(processed_dir / "test.csv")
    X_train = train[FEATURE_COLS].values
    y_train = train["target"].values.astype(int)
    X_cal = cal[FEATURE_COLS].values
    y_cal = cal["target"].values.astype(int)
    X_test = test[FEATURE_COLS].values
    y_test = test["target"].values.astype(int)

    model = ConformalXGBoost.load(models_dir)

    results: dict[str, Any] = {}

    # Section A — per-alpha coverage and set sizes on the test split.
    results["coverage"] = compute_per_alpha_coverage(model, X_test, y_test, alphas)
    plot_coverage_bar(results["coverage"], figures_dir / "coverage_guarantee.png")
    plot_set_sizes_hist(model, X_test, alphas, figures_dir / "set_sizes.png")

    # Section B — LAC vs APS vs CV+ method comparison (RAPS skipped per C35).
    results["method_comparison"] = compare_methods(
        X_train=X_train,
        y_train=y_train,
        X_cal=X_cal,
        y_cal=y_cal,
        X_test=X_test,
        y_test=y_test,
        alphas=alphas,
        xgb_params=cfg["model"]["xgb"],
        seed=int(cfg["data"]["random_seed"]),
    )
    plot_method_comparison(
        results["method_comparison"],
        figures_dir / "method_comparison_set_sizes.png",
    )

    # Persist — merge into existing reports/results.json so baseline_xgb stays.
    out_path = reports_dir / "results.json"
    if out_path.exists():
        existing = json.loads(out_path.read_text())
    else:
        existing = {}
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2, default=float))
    logger.info("Wrote evaluation results to %s", out_path)

    # MLflow — log metrics + every generated figure.
    mlflow.set_experiment(cfg["training"]["mlflow_experiment"])
    with mlflow.start_run(run_name="full_evaluation"):
        for alpha, m in results["coverage"].items():
            mlflow.log_metric(f"test_coverage_{alpha}", m["empirical_coverage"])
            mlflow.log_metric(f"test_mean_set_size_{alpha}", m["mean_set_size"])
        for fig in figures_dir.glob("*.png"):
            mlflow.log_artifact(str(fig))

    return existing


def main(config_path: str) -> None:
    """Load YAML config and dispatch to run_evaluation."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    run_evaluation(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
