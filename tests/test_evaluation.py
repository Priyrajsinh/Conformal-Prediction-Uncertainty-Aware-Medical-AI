"""Tests for the Day 4 evaluation suite — bootstraps a hermetic pipeline.

The fixture re-runs Day 2 split + scaling and Day 3 training in tmp_path so
the test does not depend on data/processed/* (gitignored, DVC-tracked) or
models/* (gitignored). This mirrors test_training.py and works the same way
locally + in CI.

Subsequent PRs will extend this file with method-comparison, Mondrian, ECE,
DCA, and selective-classification cases. PR1 ships only the coverage cases.
"""

import json
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from src.data.dataset import load_heart_disease
from src.data.preprocessing import (
    FEATURE_COLS,
    fit_scaler,
    save_splits,
    three_way_split,
)
from src.evaluation.evaluate import run_evaluation
from src.training.train import run_training


@pytest.fixture(scope="module")
def cfg_in_tmp(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Bootstrap a tmp data/processed + tmp models + tmp reports tree.

    Loads the raw UCI CSV (the one CI downloads in the workflow), runs the
    three-way stratified split, fits a StandardScaler on the train split,
    writes train/cal/test back to tmp, and points cfg paths at tmp.
    """
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    tmp_root = tmp_path_factory.mktemp("day4")
    processed_dir = tmp_root / "data" / "processed"
    models_dir = tmp_root / "models"
    reports_dir = tmp_root / "reports"
    figures_dir = reports_dir / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

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
    cfg["paths"]["reports_dir"] = str(reports_dir)
    cfg["paths"]["figures_dir"] = str(figures_dir)
    cfg["training"]["mlflow_experiment"] = "p2_test_evaluation"
    return cfg


@pytest.fixture(scope="module")
def evaluation_results(cfg_in_tmp: dict) -> dict:
    """Train + calibrate, then run the evaluation pipeline once per module."""
    run_training(cfg_in_tmp)
    return run_evaluation(cfg_in_tmp)


class TestPerAlphaCoverage:
    """Per-alpha empirical coverage and set-size summary on the test split."""

    def test_per_alpha_coverage_shape(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """results['coverage'] has one entry per configured alpha."""
        cov = evaluation_results["coverage"]
        for alpha in cfg_in_tmp["training"]["alphas"]:
            assert str(alpha) in cov
            entry = cov[str(alpha)]
            assert "empirical_coverage" in entry
            assert "mean_set_size" in entry
            assert 0.0 <= entry["empirical_coverage"] <= 1.0

    def test_coverage_meets_target_per_alpha(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """C37: empirical coverage >= 1 - alpha - tolerance for every alpha."""
        for alpha in cfg_in_tmp["training"]["alphas"]:
            cov = evaluation_results["coverage"][str(alpha)]["empirical_coverage"]
            assert (
                cov >= 1 - float(alpha) - 0.05
            ), f"alpha={alpha}: coverage {cov:.3f} below target"


class TestMethodComparison:
    """Split CP vs CV+ (k=5, 10) shape + coverage guarantees.

    APS/RAPS scores are excluded — MAPIE 1.x rejects them at runtime for binary
    targets (CLAUDE.md C35 + ValueError raised by check_target). We vary the
    splitting strategy instead, all using the LAC conformity score.
    """

    def test_method_comparison_three_methods(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """results['method_comparison'] has split_lac, cv_plus_5, cv_plus_10."""
        mc = evaluation_results["method_comparison"]
        assert set(mc.keys()) == {"split_lac", "cv_plus_5", "cv_plus_10"}
        for method in mc:
            for alpha in cfg_in_tmp["training"]["alphas"]:
                entry = mc[method][str(alpha)]
                assert "empirical_coverage" in entry
                assert "mean_set_size" in entry

    def test_method_comparison_meets_coverage(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """Every method clears the 1-alpha guarantee within tolerance."""
        mc = evaluation_results["method_comparison"]
        for method, per_alpha in mc.items():
            for alpha_str, entry in per_alpha.items():
                cov = entry["empirical_coverage"]
                assert (
                    cov >= 1 - float(alpha_str) - 0.05
                ), f"method={method} alpha={alpha_str} coverage={cov:.3f}"


class TestMondrianGroupCoverage:
    """Per-subgroup coverage and parity statistical tests."""

    def test_group_coverage_keys(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """The five reportable subgroups appear in results['group_coverage']."""
        gc = evaluation_results["group_coverage"]
        for key in ("sex_0", "sex_1", "age_lt_50", "age_50_64", "age_65p"):
            assert key in gc
            assert "n" in gc[key]
            assert "empirical_coverage" in gc[key]

    def test_parity_test_returns_pvalue(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """parity dict has a finite sex_p_value in [0, 1]."""
        parity = evaluation_results["group_coverage"]["parity"]
        assert "sex_p_value" in parity
        p = parity["sex_p_value"]
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


class TestExpectedCalibrationError:
    """ECE value lies in the unit interval and gets persisted."""

    def test_ece_in_unit_interval(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """ECE in [0, 1] (it's a weighted average of |frac_pos - mean_pred|)."""
        ece = evaluation_results["ece"]
        assert "value" in ece
        assert 0.0 <= float(ece["value"]) <= 1.0
        assert ece["n_bins"] == 10


class TestDecisionCurveAnalysis:
    """Net-benefit curves: shape and consistency."""

    def test_dca_three_curves(self, cfg_in_tmp: dict, evaluation_results: dict) -> None:
        """results['dca'] has model + treat_all + treat_none, all same length."""
        dca = evaluation_results["dca"]
        for key in ("model", "treat_all", "treat_none", "thresholds"):
            assert key in dca
        n = len(dca["thresholds"])
        assert len(dca["model"]) == n
        assert len(dca["treat_all"]) == n
        assert len(dca["treat_none"]) == n
        # treat_none is by definition zero everywhere.
        assert all(v == 0.0 for v in dca["treat_none"])


class TestArtefacts:
    """Figure artefacts and the merged results.json file."""

    def test_coverage_bar_written(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """coverage_guarantee.png is written to the figures directory."""
        path = Path(cfg_in_tmp["paths"]["figures_dir"], "coverage_guarantee.png")
        assert path.exists() and path.stat().st_size > 0

    def test_set_sizes_hist_written(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """set_sizes.png is written to the figures directory."""
        path = Path(cfg_in_tmp["paths"]["figures_dir"], "set_sizes.png")
        assert path.exists() and path.stat().st_size > 0

    def test_method_comparison_chart_written(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """method_comparison_set_sizes.png is written to the figures directory."""
        path = Path(
            cfg_in_tmp["paths"]["figures_dir"], "method_comparison_set_sizes.png"
        )
        assert path.exists() and path.stat().st_size > 0

    def test_group_coverage_chart_written(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """group_coverage.png is written to the figures directory."""
        path = Path(cfg_in_tmp["paths"]["figures_dir"], "group_coverage.png")
        assert path.exists() and path.stat().st_size > 0

    def test_calibration_chart_written(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """calibration.png is written to the figures directory."""
        path = Path(cfg_in_tmp["paths"]["figures_dir"], "calibration.png")
        assert path.exists() and path.stat().st_size > 0

    def test_dca_chart_written(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """dca_net_benefit.png is written to the figures directory."""
        path = Path(cfg_in_tmp["paths"]["figures_dir"], "dca_net_benefit.png")
        assert path.exists() and path.stat().st_size > 0

    def test_results_json_persists_top_level_keys(
        self, cfg_in_tmp: dict, evaluation_results: dict
    ) -> None:
        """reports/results.json contains every section key emitted so far."""
        path = Path(cfg_in_tmp["paths"]["reports_dir"], "results.json")
        assert path.exists()
        payload = json.loads(path.read_text())
        for key in ("coverage", "method_comparison", "group_coverage", "ece", "dca"):
            assert key in payload, f"missing {key}"
