"""Selective classification accuracy versus abstain-rate sweep (rule C).

Conformal prediction supplies an *abstention signal for free*: when the
prediction set has more than one label the model is admitting it cannot
commit, and we should defer to a human reviewer. Sweeping alpha lets the
operator trade off coverage versus throughput:

* Low alpha (e.g. 0.01) → bigger sets → more abstentions, but the
  predictions we DO make are nearly certain.
* High alpha (e.g. 0.50) → smaller sets, fewer abstentions, but the
  remaining decisions are riskier.

The classical risk-coverage tradeoff says: as abstain rate increases, the
accuracy of the predictions we still issue is non-decreasing. The plot in
``reports/figures/selective_accuracy.png`` makes that curve explicit.
"""

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from mapie.classification import SplitConformalClassifier

from src.logger import get_logger
from src.models.model import ConformalXGBoost

logger = get_logger(__name__)

SelectiveResults = Dict[str, list]


def selective_accuracy_curve(
    model: ConformalXGBoost,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alpha_sweep: np.ndarray,
) -> SelectiveResults:
    """Sweep alpha; for each alpha report (abstain_rate, accuracy_on_predicted).

    Builds a single ``SplitConformalClassifier`` with one confidence level per
    sweep alpha so the calibration cost is paid once. ``model.xgb`` is reused
    (prefit=True) so we never re-fit the underlying classifier. Abstention
    rule: ``|set| != 1`` (also abstains on empty sets — those signal a
    failure to commit at all).
    """
    sweep = np.asarray(alpha_sweep).astype(float)
    n_cal = int(len(y_cal))
    # MAPIE requires 1/alpha < n_cal AND 1/(1-alpha) < n_cal. Clamp the sweep
    # to the data-feasible range; with n_cal=60 the lower bound is ~0.017.
    safe_lo = 1.0 / max(n_cal - 1, 1) + 1e-6
    safe_hi = 1.0 - safe_lo
    sweep = sweep[(sweep > safe_lo) & (sweep < safe_hi)]
    if sweep.size == 0:
        raise ValueError(
            f"alpha sweep is empty after clamping to ({safe_lo:.4f}, "
            f"{safe_hi:.4f}); n_cal={n_cal}"
        )
    confidence_levels = [1.0 - a for a in sweep]
    scc = SplitConformalClassifier(
        estimator=model.xgb,
        confidence_level=confidence_levels,
        conformity_score="lac",
        prefit=True,
    )
    scc.conformalize(X_cal, y_cal)
    y_pred, y_set_all = scc.predict_set(X_test)
    y_test_arr = np.asarray(y_test).astype(int)
    n = int(len(y_test_arr))

    abstain_rates: list[float] = []
    accuracies: list[float] = []
    for idx, alpha in enumerate(sweep):
        y_set = y_set_all[:, :, idx]
        sizes = y_set.sum(axis=1)
        abstain_mask = sizes != 1
        n_abstain = int(abstain_mask.sum())
        n_predicted = n - n_abstain
        if n_predicted == 0:
            accuracy = float("nan")
        else:
            kept_pred = np.asarray(y_pred)[~abstain_mask]
            kept_true = y_test_arr[~abstain_mask]
            accuracy = float((kept_pred == kept_true).sum() / n_predicted)
        abstain_rate = float(n_abstain / n)
        abstain_rates.append(abstain_rate)
        accuracies.append(accuracy)
        logger.info(
            "selective alpha=%.3f abstain=%.3f accuracy=%s",
            alpha,
            abstain_rate,
            f"{accuracy:.3f}" if not np.isnan(accuracy) else "nan",
        )

    return {
        "alphas": sweep.tolist(),
        "abstain_rates": abstain_rates,
        "accuracies": accuracies,
    }


def plot_selective_curve(results: SelectiveResults, out_path: Path) -> None:
    """Render accuracy versus abstain-rate, sorted by abstain rate ascending."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pairs = sorted(zip(results["abstain_rates"], results["accuracies"]))
    abstain = [p[0] for p in pairs]
    acc = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        abstain,
        acc,
        marker="o",
        linewidth=2,
        color="#4f46e5",
        label="accuracy when not abstaining",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Abstain rate")
    ax.set_ylabel("Accuracy on predicted subset")
    ax.set_title("Selective classification: accuracy vs abstain rate")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote selective-classification chart to %s", out_path)
