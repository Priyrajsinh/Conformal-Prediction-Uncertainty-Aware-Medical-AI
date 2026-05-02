"""ConformalXGBoost: XGBoost wrapped by MAPIE 1.x SplitConformalClassifier.

Calibration uses the LAC (Least Ambiguous set-valued Classifier) conformity
score on a held-out cal split — MAPIE 1.x mandates LAC for binary targets and
reserves RAPS for multiclass (n_classes >= 3). See CLAUDE.md C35.
"""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from mapie.classification import SplitConformalClassifier
from xgboost import XGBClassifier

from src.exceptions import ModelNotFoundError, PredictionError
from src.logger import get_logger
from src.models.base import BaseMLModel

logger = get_logger(__name__)


class ConformalXGBoost(BaseMLModel):
    """XGBoost classifier wrapped by a MAPIE SplitConformalClassifier.

    Lifecycle: ``fit(X_train, y_train)`` -> ``calibrate(X_cal, y_cal, alphas)``
    -> ``safe_predict(X, alpha)``. Calibration happens on the cal split only
    (rule C35). ``safe_predict`` adds NaN/inf/shape guards (rule C36).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        """Construct an XGBoost estimator with the given hyperparameters."""
        self.xgb = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
            eval_metric="logloss",
        )
        self.scc: SplitConformalClassifier | None = None
        self.alphas: list[float] = []
        self._fitted = False
        self._calibrated = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalXGBoost":
        """Train the underlying XGBoost classifier on (X, y)."""
        self.xgb.fit(X, y)
        self._fitted = True
        return self

    def calibrate(
        self,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        alphas: list[float],
    ) -> "ConformalXGBoost":
        """Conformalize the fitted XGBoost on the cal split for each requested alpha.

        confidence_level is set to ``[1 - a for a in alphas]`` so a single
        ``predict_set`` call returns sets for every alpha at once. The conformity
        score is fixed to ``'lac'`` (the only valid choice for binary targets in
        MAPIE 1.x; rule C35).
        """
        if not self._fitted:
            raise PredictionError("Cannot calibrate before fit().")
        confidence_levels = [1.0 - a for a in alphas]
        self.scc = SplitConformalClassifier(
            estimator=self.xgb,
            confidence_level=confidence_levels,
            conformity_score="lac",
            prefit=True,
        )
        self.scc.conformalize(X_cal, y_cal)
        self.alphas = list(alphas)
        self._calibrated = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the underlying XGBoost hard predictions for *X*."""
        if not self._fitted:
            raise PredictionError("Model not fitted.")
        return np.asarray(self.xgb.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return the underlying XGBoost class probabilities for *X*."""
        if not self._fitted:
            raise PredictionError("Model not fitted.")
        return np.asarray(self.xgb.predict_proba(X))

    def safe_predict(
        self, X: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (hard predictions, prediction-set indicator) with input guards.

        Rule C36: every call to MAPIE's ``predict_set`` must be wrapped by NaN/inf
        and shape checks so silent data-quality failures cannot reach production.
        Logs input feature stats before calling MAPIE.
        """
        if not self._calibrated or self.scc is None:
            raise PredictionError("Model not calibrated.")
        if X.ndim != 2:
            raise PredictionError(f"Expected 2-D X, got shape {X.shape}.")
        if np.isnan(X).any() or np.isinf(X).any():
            raise PredictionError("Input contains NaN or inf.")
        if alpha not in self.alphas:
            raise PredictionError(
                f"alpha={alpha} not in calibrated alphas {self.alphas}."
            )
        logger.info(
            "safe_predict: n=%d mean=%.3f std=%.3f alpha=%.2f",
            X.shape[0],
            float(X.mean()),
            float(X.std()),
            alpha,
        )
        idx = self.alphas.index(alpha)
        y_pred, y_set_all = self.scc.predict_set(X)
        # y_set_all shape: (n_samples, n_classes, n_alphas)
        y_set = y_set_all[:, :, idx]
        return np.asarray(y_pred), np.asarray(y_set)

    def save(self, path: Path) -> None:
        """Persist xgb + scc + meta to ``path/`` (creates dir if missing)."""
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.xgb, path / "xgb.joblib")
        if self.scc is not None:
            joblib.dump(self.scc, path / "scc.joblib")
        meta: dict[str, Any] = {
            "alphas": self.alphas,
            "fitted": self._fitted,
            "calibrated": self._calibrated,
        }
        joblib.dump(meta, path / "meta.joblib")
        logger.info("Saved ConformalXGBoost to %s", path)

    @classmethod
    def load(cls, path: Path) -> "ConformalXGBoost":
        """Reconstruct a ConformalXGBoost from a previously-saved directory."""
        if not (path / "xgb.joblib").exists():
            raise ModelNotFoundError(f"No xgb.joblib in {path}")
        obj = cls()
        obj.xgb = joblib.load(path / "xgb.joblib")
        obj._fitted = True
        if (path / "scc.joblib").exists():
            obj.scc = joblib.load(path / "scc.joblib")
            obj._calibrated = True
        if (path / "meta.joblib").exists():
            meta = joblib.load(path / "meta.joblib")
            obj.alphas = meta.get("alphas", [])
        return obj
