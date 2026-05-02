"""Smoke tests for src/training/train.py — full pipeline + MLflow assertions."""

import json
from pathlib import Path

import mlflow
import pytest
import yaml  # type: ignore[import-untyped]

from src.data.dataset import load_heart_disease
from src.data.preprocessing import (
    FEATURE_COLS,
    fit_scaler,
    save_splits,
    three_way_split,
)
from src.training.train import run_training


@pytest.fixture(scope="module")
def cfg_in_tmp(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Bootstrap processed splits in tmp, redirect cfg paths + experiment to tmp.

    data/processed/*.csv is gitignored (DVC-tracked), so CI does not have those
    files even after the raw curl. Regenerate them in tmp from the raw CSV so
    this test is hermetic across local + CI runs.
    """
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    tmp_root = tmp_path_factory.mktemp("day3")
    processed_dir = tmp_root / "data" / "processed"
    models_dir = tmp_root / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    df = load_heart_disease(cfg["data"]["raw_path"])
    train, cal, test = three_way_split(
        df,
        cfg["data"]["train_pct"],
        cfg["data"]["cal_pct"],
        cfg["data"]["test_pct"],
        stratify_col="target",
        seed=cfg["data"]["random_seed"],
    )
    train_s, cal_s, test_s = fit_scaler(
        train, cal, test, FEATURE_COLS, models_dir / "scaler.joblib"
    )
    save_splits(train_s, cal_s, test_s, processed_dir)

    cfg["data"]["processed_dir"] = str(processed_dir)
    cfg["paths"]["models_dir"] = str(models_dir)
    cfg["training"]["mlflow_experiment"] = "p2_test_smoke"
    return cfg


@pytest.fixture(scope="module")
def trained(cfg_in_tmp: dict) -> dict[str, dict[str, float]]:
    """Run the full training pipeline once per module and return coverage."""
    return run_training(cfg_in_tmp)


class TestTrainingArtefacts:
    """Files written by run_training to the configured models_dir."""

    def test_xgb_artefact_exists(self, cfg_in_tmp: dict, trained: dict) -> None:
        """xgb.joblib written to models_dir."""
        assert Path(cfg_in_tmp["paths"]["models_dir"], "xgb.joblib").exists()

    def test_scc_artefact_exists(self, cfg_in_tmp: dict, trained: dict) -> None:
        """scc.joblib (the conformalized SplitConformalClassifier) is written."""
        assert Path(cfg_in_tmp["paths"]["models_dir"], "scc.joblib").exists()

    def test_meta_artefact_exists(self, cfg_in_tmp: dict, trained: dict) -> None:
        """meta.joblib (alphas + state flags) is written."""
        assert Path(cfg_in_tmp["paths"]["models_dir"], "meta.joblib").exists()

    def test_training_stats_exists(self, cfg_in_tmp: dict, trained: dict) -> None:
        """training_stats.json is written for Day 5 skew checks."""
        assert Path(cfg_in_tmp["paths"]["models_dir"], "training_stats.json").exists()


class TestCalibrationMetadata:
    """Provenance file consumed by Day 8 anti-leakage test."""

    def test_calibration_metadata_written(
        self, cfg_in_tmp: dict, trained: dict
    ) -> None:
        """calibration_metadata.json exists with the correct provenance."""
        path = Path(cfg_in_tmp["paths"]["models_dir"], "calibration_metadata.json")
        assert path.exists()
        with open(path) as f:
            meta = json.load(f)
        assert meta["split_used"] == "cal"
        assert meta["conformity_score"] == "lac"
        assert meta["prefit"] is True
        assert meta["alphas"] == cfg_in_tmp["training"]["alphas"]


class TestCoverageGuarantee:
    """Rule C37: empirical coverage must satisfy >= 1 - alpha - tolerance."""

    def test_coverage_meets_target_per_alpha(
        self, cfg_in_tmp: dict, trained: dict
    ) -> None:
        """For every configured alpha, empirical coverage >= 1 - alpha - 0.05."""
        for alpha in cfg_in_tmp["training"]["alphas"]:
            cov = trained[str(alpha)]["empirical_coverage"]
            assert (
                cov >= 1 - alpha - 0.05
            ), f"alpha={alpha}: coverage {cov:.3f} below target"


class TestMLflowRun:
    """MLflow records the conformal_xgb_lac run with the expected metrics."""

    def test_mlflow_run_named_correctly(self, cfg_in_tmp: dict, trained: dict) -> None:
        """A run named 'conformal_xgb_lac' appears in the configured experiment."""
        client = mlflow.MlflowClient()
        exp = client.get_experiment_by_name(cfg_in_tmp["training"]["mlflow_experiment"])
        assert exp is not None
        runs = client.search_runs(
            [exp.experiment_id],
            filter_string="tags.mlflow.runName = 'conformal_xgb_lac'",
        )
        assert len(runs) >= 1
