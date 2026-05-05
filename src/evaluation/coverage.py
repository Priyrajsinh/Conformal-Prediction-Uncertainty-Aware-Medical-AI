"""Per-alpha coverage and prediction-set summary statistics on the test split.

The coverage check is the central conformal-prediction guarantee: empirical
coverage must be at least 1 - alpha (rule C37). This module computes that
metric on the held-out test split (never the cal split, which is reserved for
calibration; rule C35) and ships matplotlib bar/histogram plots for the
report figures directory.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.logger import get_logger
from src.models.model import ConformalXGBoost

logger = get_logger(__name__)


def compute_per_alpha_coverage(
    model: ConformalXGBoost,
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float],
) -> dict[str, dict[str, float]]:
    """Compute empirical coverage and set-size stats on (X, y) for each alpha.

    For every alpha the calibrated model exposes, run safe_predict, score the
    indicator at the true label, and report empirical coverage plus mean,
    median, and empty-set count. Returns a dict keyed by str(alpha) so the
    output is JSON-serialisable for ``reports/results.json``.
    """
    results: dict[str, dict[str, float]] = {}
    n = int(len(y))
    for alpha in alphas:
        _, y_set = model.safe_predict(X, alpha=float(alpha))
        covered = float(y_set[np.arange(n), y].sum() / n)
        sizes = y_set.sum(axis=1)
        results[str(alpha)] = {
            "empirical_coverage": covered,
            "mean_set_size": float(sizes.mean()),
            "median_set_size": float(np.median(sizes)),
            "n_empty_sets": int((sizes == 0).sum()),
            "n": n,
        }
        logger.info(
            "alpha=%.2f coverage=%.3f mean_size=%.2f empty=%d",
            alpha,
            covered,
            float(sizes.mean()),
            int((sizes == 0).sum()),
        )
    return results


def plot_coverage_bar(
    results: dict[str, dict[str, float]],
    out_path: Path,
) -> None:
    """Render a bar chart of empirical coverage per alpha with 1-alpha refs."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    alphas = sorted(results.keys(), key=float)
    cov = [results[a]["empirical_coverage"] for a in alphas]
    targets = [1.0 - float(a) for a in alphas]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(alphas))
    ax.bar(x, cov, width=0.55, color="#4f46e5", label="Empirical coverage")
    for xi, t in zip(x, targets):
        ax.hlines(
            t, xi - 0.3, xi + 0.3, colors="#dc2626", linestyles="dashed", linewidth=2
        )
    ax.set_xticks(x)
    ax.set_xticklabels([f"alpha={a}" for a in alphas])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage")
    ax.set_title("Empirical coverage vs target (1 - alpha)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote coverage bar chart to %s", out_path)


def plot_set_sizes_hist(
    model: ConformalXGBoost,
    X: np.ndarray,
    alphas: list[float],
    out_path: Path,
) -> None:
    """Render one histogram panel per alpha showing prediction-set sizes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_alphas = len(alphas)
    fig, axes = plt.subplots(1, n_alphas, figsize=(4 * n_alphas, 4), sharey=True)
    if n_alphas == 1:
        axes = [axes]
    for ax, alpha in zip(axes, alphas):
        _, y_set = model.safe_predict(X, alpha=float(alpha))
        sizes = y_set.sum(axis=1)
        ax.hist(
            sizes,
            bins=np.arange(-0.5, sizes.max() + 1.5),
            color="#7c3aed",
            edgecolor="white",
        )
        ax.set_title(f"alpha = {alpha}")
        ax.set_xlabel("Prediction-set size")
    axes[0].set_ylabel("Count")
    fig.suptitle("Prediction-set size distribution by alpha")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote set-size histograms to %s", out_path)
