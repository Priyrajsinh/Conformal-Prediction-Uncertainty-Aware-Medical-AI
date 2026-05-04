"""Expected Calibration Error and reliability diagram (rule U2).

Conformal prediction guarantees set-valued coverage but does NOT guarantee
that the underlying probability scores are calibrated. ECE measures the
weighted gap between predicted probabilities and empirical positive
fractions across uniform-width bins; the reliability diagram visualises
the same numbers against the diagonal `y = x` reference line.

A lower ECE means the probability scores are usable as actual
probabilities (e.g. for downstream decision curves), independent of the
conformal set the patient receives.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

from src.logger import get_logger

logger = get_logger(__name__)


def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute ECE plus the (frac_pos, mean_pred) curve from sklearn.

    Bins are equal-width on [0, 1]. ``calibration_curve`` drops empty bins,
    so we weight each surviving bin by its observed share of the total
    sample mass before summing the absolute deviations.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    frac_pos, mean_pred = calibration_curve(
        y_true_arr, y_proba_arr, n_bins=n_bins, strategy="uniform"
    )
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.digitize(y_proba_arr, bin_edges[1:-1])
    bin_counts = np.bincount(bin_idx, minlength=n_bins).astype(float)
    total = max(float(bin_counts.sum()), 1.0)
    weights = bin_counts / total
    surviving_bin_ids = np.unique(bin_idx)
    surviving_bin_ids = surviving_bin_ids[surviving_bin_ids < n_bins]
    n_keep = min(len(surviving_bin_ids), len(frac_pos))
    aligned_weights = weights[surviving_bin_ids][:n_keep]
    ece = float(
        np.sum(np.abs(frac_pos[:n_keep] - mean_pred[:n_keep]) * aligned_weights)
    )
    logger.info("ECE=%.4f n_bins=%d n_samples=%d", ece, n_bins, len(y_true_arr))
    return ece, frac_pos, mean_pred


def plot_reliability_diagram(
    frac_pos: np.ndarray,
    mean_pred: np.ndarray,
    ece: float,
    out_path: Path,
) -> None:
    """Render reliability diagram with diagonal reference and ECE in the title."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#dc2626", label="perfect calibration")
    ax.plot(
        mean_pred,
        frac_pos,
        marker="o",
        linewidth=2,
        color="#4f46e5",
        label="model",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability (per bin)")
    ax.set_ylabel("Empirical positive fraction (per bin)")
    ax.set_title(f"Reliability diagram - ECE = {ece:.4f}")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote reliability diagram to %s", out_path)
