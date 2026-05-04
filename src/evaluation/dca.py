"""Decision Curve Analysis (rule A).

Standard accuracy/AUC scores treat false positives and false negatives as
equally costly, which is wrong for medical screening: a missed heart-disease
case (FN) is far worse than a false alarm (FP). DCA expresses both kinds of
error in the same currency by weighting them through the threshold of
"willingness to treat" t:

    net_benefit(t) = TP / N - FP / N * (t / (1 - t))

The model is reported alongside two reference strategies:

* **treat-all** — flag every patient. ``net_benefit = prev - (1-prev) * t/(1-t)``.
* **treat-none** — flag nobody. ``net_benefit = 0``.

A model is clinically useful at threshold t if its curve is above both
references. The DCA chart in ``reports/figures/dca_net_benefit.png`` shows
that range at a glance.
"""

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

DCAResults = Dict[str, list]


def decision_curve_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> DCAResults:
    """Return net-benefit curves for the model, treat-all, and treat-none.

    All three curves share the same threshold grid so the plotting layer can
    line them up without re-keying. ``thresholds`` defaults to a 99-point
    sweep on (0.01, 0.99); thresholds at the boundary cause a divide-by-zero
    in ``t / (1 - t)`` and are explicitly excluded.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    n = int(len(y_true_arr))
    prev = float(y_true_arr.mean())

    if thresholds is None:
        thresholds = np.arange(0.01, 0.99 + 1e-9, 0.01)
    t_arr = np.asarray(thresholds).astype(float)

    model_nb: list[float] = []
    treat_all_nb: list[float] = []
    treat_none_nb: list[float] = []
    for t in t_arr:
        odds = t / max(1.0 - t, 1e-9)
        y_hat = (y_proba_arr >= t).astype(int)
        tp = float(((y_hat == 1) & (y_true_arr == 1)).sum())
        fp = float(((y_hat == 1) & (y_true_arr == 0)).sum())
        model_nb.append(tp / n - (fp / n) * odds)
        treat_all_nb.append(prev - (1.0 - prev) * odds)
        treat_none_nb.append(0.0)

    logger.info(
        "DCA: thresholds=%d prev=%.3f model peak=%.3f",
        len(t_arr),
        prev,
        max(model_nb) if model_nb else 0.0,
    )
    return {
        "thresholds": t_arr.tolist(),
        "model": model_nb,
        "treat_all": treat_all_nb,
        "treat_none": treat_none_nb,
        "prevalence": [prev],
    }


def plot_dca(results: DCAResults, out_path: Path) -> None:
    """Render a line chart of the model versus the two reference strategies."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.asarray(results["thresholds"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, results["model"], color="#4f46e5", linewidth=2.5, label="model")
    ax.plot(
        t,
        results["treat_all"],
        color="#0ea5e9",
        linewidth=1.5,
        linestyle="--",
        label="treat all",
    )
    ax.plot(
        t,
        results["treat_none"],
        color="#dc2626",
        linewidth=1.5,
        linestyle=":",
        label="treat none",
    )
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Threshold probability (t)")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis")
    ax.set_ylim(min(-0.05, min(results["model"])) - 0.02, max(results["model"]) + 0.02)
    ax.legend(loc="upper right")
    ax.grid(linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote DCA chart to %s", out_path)
