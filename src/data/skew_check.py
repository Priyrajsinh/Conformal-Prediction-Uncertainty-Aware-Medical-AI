"""Population Stability Index (PSI) based training-serving skew detection."""

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
