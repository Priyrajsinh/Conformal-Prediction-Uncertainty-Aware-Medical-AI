"""Population Stability Index (PSI) based training-serving skew detection."""

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.exceptions import SkewError
from src.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_THRESHOLD = 0.20
_DEFAULT_BUCKETS = 10


def compute_psi(
    expected: pd.Series,
    actual: pd.Series,
    buckets: int = _DEFAULT_BUCKETS,
) -> float:
    """Compute Population Stability Index between *expected* and *actual* distributions.

    PSI < 0.10 = no significant change; 0.10-0.20 = moderate; > 0.20 = significant.
    """
    min_val = float(min(expected.min(), actual.min()))
    max_val = float(max(expected.max(), actual.max()))
    bins = np.linspace(min_val, max_val, buckets + 1)

    expected_counts, _ = np.histogram(expected, bins=bins)
    actual_counts, _ = np.histogram(actual, bins=bins)

    expected_pct = (expected_counts + 1e-6) / len(expected)
    actual_pct = (actual_counts + 1e-6) / len(actual)

    psi = float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))
    return psi


def check_skew(
    train_df: pd.DataFrame,
    serving_df: pd.DataFrame,
    features: Sequence[str],
    threshold: float = _DEFAULT_THRESHOLD,
) -> dict[str, float]:
    """Compute PSI for each feature; raise SkewError if any exceeds *threshold*."""
    results: dict[str, float] = {}
    violations: list[str] = []

    for feat in features:
        psi = compute_psi(train_df[feat], serving_df[feat])
        results[feat] = psi
        if psi > threshold:
            violations.append(f"{feat}={psi:.4f}")

    if violations:
        raise SkewError(
            f"Skew detected: {', '.join(violations)} (threshold={threshold})"
        )

    logger.info("Skew check passed for %d features", len(features))
    return results


def save_training_stats(
    X: np.ndarray,
    path: Path,
    feature_names: list[str] | None = None,
) -> None:
    """Persist per-feature mean/std/min/max of training X to JSON for Day 5 skew checks.

    If *feature_names* is None, generic ``feature_{i}`` keys are used. Output is
    written as ``{name: {mean, std, min, max}}`` so it can be re-loaded as a
    pandas DataFrame and validated by pandera (rule C23).
    """
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D X, got shape {X.shape}.")
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    if len(feature_names) != X.shape[1]:
        raise ValueError(
            f"feature_names length {len(feature_names)} != X.shape[1]={X.shape[1]}."
        )
    stats = {
        name: {
            "mean": float(X[:, i].mean()),
            "std": float(X[:, i].std()),
            "min": float(X[:, i].min()),
            "max": float(X[:, i].max()),
        }
        for i, name in enumerate(feature_names)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved training stats for %d features -> %s", len(feature_names), path)


def check_serving_skew(
    X: np.ndarray,
    stats_path: Path,
    z_threshold: float = 3.0,
) -> dict[str, bool]:
    """Return per-feature out-of-distribution flags for a single serving row.

    Loads per-feature {mean, std} from *stats_path* (written by
    ``save_training_stats``) and returns True for any feature whose z-score
    exceeds *z_threshold*. Designed for single-row inference (shape (1, n_features)).
    """
    with open(stats_path) as f:
        stats: dict[str, dict[str, float]] = json.load(f)

    feature_names = list(stats.keys())
    result: dict[str, bool] = {}
    for i, name in enumerate(feature_names):
        if i >= X.shape[1]:
            break
        s = stats[name]
        std = s["std"] if s["std"] > 1e-9 else 1.0
        z = abs((float(X[0, i]) - s["mean"]) / std)
        result[name] = z > z_threshold
    return result
