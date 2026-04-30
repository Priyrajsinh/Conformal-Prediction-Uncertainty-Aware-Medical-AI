"""Abstract base class defining the fit/predict/save/load interface for all models."""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd


class BaseMLModel(ABC):
    """Interface contract for every ML model in the conformal prediction pipeline."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseMLModel":
        """Fit the model on *X* and *y* and return self."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard class predictions for *X*."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probability matrix for *X* (shape: n_samples × n_classes)."""

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist the fitted model artefact to *path*."""

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseMLModel":
        """Load and return a persisted model from *path*."""
