"""Unit tests for src/models/model.py — ConformalXGBoost lifecycle + guards."""

from pathlib import Path

import numpy as np
import pytest

from src.exceptions import ModelNotFoundError, PredictionError
from src.models.model import ConformalXGBoost


@pytest.fixture(scope="module")
def synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_train, y_train, X_cal, y_cal) for a separable binary task."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(200, 5))
    y_train = (X_train[:, 0] + rng.normal(0, 0.3, 200) > 0).astype(int)
    X_cal = rng.normal(size=(80, 5))
    y_cal = (X_cal[:, 0] + rng.normal(0, 0.3, 80) > 0).astype(int)
    return X_train, y_train, X_cal, y_cal


@pytest.fixture(scope="module")
def calibrated_model(
    synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> ConformalXGBoost:
    """Return a ConformalXGBoost fit on train and calibrated on cal."""
    X_train, y_train, X_cal, y_cal = synthetic_data
    model = ConformalXGBoost(n_estimators=20, max_depth=3)
    model.fit(X_train, y_train).calibrate(X_cal, y_cal, alphas=[0.05, 0.1])
    return model


class TestLifecycleGuards:
    """Tests for fit/calibrate/safe_predict ordering and input guards."""

    def test_calibrate_before_fit_raises(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """calibrate() before fit() must raise PredictionError."""
        _, _, X_cal, y_cal = synthetic_data
        model = ConformalXGBoost(n_estimators=10)
        with pytest.raises(PredictionError, match="Cannot calibrate"):
            model.calibrate(X_cal, y_cal, alphas=[0.1])

    def test_safe_predict_before_calibrate_raises(
        self,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """safe_predict() before calibrate() must raise PredictionError."""
        X_train, y_train, _, _ = synthetic_data
        model = ConformalXGBoost(n_estimators=10).fit(X_train, y_train)
        with pytest.raises(PredictionError, match="not calibrated"):
            model.safe_predict(X_train, alpha=0.1)

    def test_predict_before_fit_raises(self) -> None:
        """predict() before fit() must raise PredictionError."""
        model = ConformalXGBoost(n_estimators=10)
        with pytest.raises(PredictionError, match="not fitted"):
            model.predict(np.zeros((5, 5)))

    def test_predict_proba_before_fit_raises(self) -> None:
        """predict_proba() before fit() must raise PredictionError."""
        model = ConformalXGBoost(n_estimators=10)
        with pytest.raises(PredictionError, match="not fitted"):
            model.predict_proba(np.zeros((5, 5)))


class TestSafePredictGuards:
    """Rule C36: NaN / inf / shape guards on safe_predict."""

    def test_nan_raises(self, calibrated_model: ConformalXGBoost) -> None:
        """Input containing NaN raises PredictionError."""
        X_bad = np.zeros((3, 5))
        X_bad[0, 0] = np.nan
        with pytest.raises(PredictionError, match="NaN or inf"):
            calibrated_model.safe_predict(X_bad, alpha=0.1)

    def test_inf_raises(self, calibrated_model: ConformalXGBoost) -> None:
        """Input containing inf raises PredictionError."""
        X_bad = np.zeros((3, 5))
        X_bad[0, 0] = np.inf
        with pytest.raises(PredictionError, match="NaN or inf"):
            calibrated_model.safe_predict(X_bad, alpha=0.1)

    def test_wrong_shape_raises(self, calibrated_model: ConformalXGBoost) -> None:
        """1-D input raises PredictionError."""
        with pytest.raises(PredictionError, match="2-D"):
            calibrated_model.safe_predict(np.zeros(5), alpha=0.1)

    def test_unknown_alpha_raises(self, calibrated_model: ConformalXGBoost) -> None:
        """Asking for an alpha that wasn't calibrated raises PredictionError."""
        with pytest.raises(PredictionError, match="not in calibrated"):
            calibrated_model.safe_predict(np.zeros((3, 5)), alpha=0.99)


class TestPredictionSets:
    """Coverage and cardinality properties of prediction sets."""

    def test_set_nonempty(
        self,
        calibrated_model: ConformalXGBoost,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Every prediction set has cardinality >= 1."""
        _, _, X_cal, _ = synthetic_data
        _, y_set = calibrated_model.safe_predict(X_cal, alpha=0.05)
        assert (y_set.sum(axis=1) >= 1).all()

    def test_set_shape_two_classes(
        self,
        calibrated_model: ConformalXGBoost,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Set indicator has shape (n_samples, 2) for binary task."""
        _, _, X_cal, _ = synthetic_data
        _, y_set = calibrated_model.safe_predict(X_cal, alpha=0.05)
        assert y_set.shape == (X_cal.shape[0], 2)


class TestSerialization:
    """Save/load round-trip preserves model behavior."""

    def test_save_load_roundtrip(
        self,
        calibrated_model: ConformalXGBoost,
        synthetic_data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        tmp_path: Path,
    ) -> None:
        """predict() and safe_predict() match before and after save->load."""
        _, _, X_cal, _ = synthetic_data
        pred_before = calibrated_model.predict(X_cal)
        _, set_before = calibrated_model.safe_predict(X_cal, alpha=0.1)
        calibrated_model.save(tmp_path)
        loaded = ConformalXGBoost.load(tmp_path)
        pred_after = loaded.predict(X_cal)
        _, set_after = loaded.safe_predict(X_cal, alpha=0.1)
        np.testing.assert_array_equal(pred_before, pred_after)
        np.testing.assert_array_equal(set_before, set_after)

    def test_load_missing_xgb_raises(self, tmp_path: Path) -> None:
        """load() against an empty dir raises ModelNotFoundError."""
        with pytest.raises(ModelNotFoundError):
            ConformalXGBoost.load(tmp_path)
