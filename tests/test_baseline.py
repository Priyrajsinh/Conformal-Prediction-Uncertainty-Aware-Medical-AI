"""Tests for src/baseline/baseline.py — accuracy, results.json key."""

import json
from pathlib import Path

import pandas as pd
import pytest

from src.baseline.baseline import FEATURE_COLS, train_baseline
from src.data.dataset import load_heart_disease
from src.data.preprocessing import fit_scaler, three_way_split

RAW_PATH = Path("data/raw/heart.csv")

XGB_PARAMS: dict[str, int | float | str | bool] = {
    "n_estimators": 50,
    "max_depth": 4,
    "learning_rate": 0.05,
    "random_state": 42,
    "eval_metric": "logloss",
}


@pytest.fixture(scope="module")
def splits(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load heart data and return scaled (train, cal, test) splits."""
    tmp = tmp_path_factory.mktemp("models")
    df = load_heart_disease(RAW_PATH)
    train, cal, test = three_way_split(df, 0.6, 0.2, 0.2, "target", 42)
    train, cal, test = fit_scaler(train, cal, test, FEATURE_COLS, tmp / "scaler.joblib")
    return train, cal, test


class TestBaselineAccuracy:
    """Tests that verify model quality on real data."""

    def test_accuracy_above_threshold(
        self,
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        tmp_path: Path,
    ) -> None:
        """Baseline accuracy on the test split must exceed 0.70."""
        train, _, test = splits
        figures_dir = tmp_path / "figures"
        results_path = tmp_path / "results.json"
        train_baseline(
            train,
            test,
            figures_dir,
            results_path,
            "p2_conformal_prediction_test",
            XGB_PARAMS,
        )
        with open(results_path) as f:
            results = json.load(f)
        assert results["baseline_xgb"]["accuracy"] > 0.70

    def test_results_json_contains_baseline_key(
        self,
        splits: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
        tmp_path: Path,
    ) -> None:
        """results.json written by train_baseline must contain 'baseline_xgb' key."""
        train, _, test = splits
        results_path = tmp_path / "results.json"
        train_baseline(
            train,
            test,
            tmp_path / "figures",
            results_path,
            "p2_conformal_prediction_test",
            XGB_PARAMS,
        )
        assert results_path.exists()
        with open(results_path) as f:
            data = json.load(f)
        assert "baseline_xgb" in data
