"""Smoke tests for src/training/train.py — full pipeline + MLflow assertions."""

import json
from pathlib import Path

import mlflow
import pytest
import yaml  # type: ignore[import-untyped]

from src.training.train import run_training


@pytest.fixture(scope="module")
def cfg_in_tmp(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Load real config but redirect models_dir + experiment to a tmp scope."""
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    tmp = tmp_path_factory.mktemp("models")
    cfg["paths"]["models_dir"] = str(tmp)
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
