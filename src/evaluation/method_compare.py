"""Compare conformal methods on the test split: Split CP vs CV+ (5) vs CV+ (10).

CLAUDE.md C35 forbids any conformity score other than LAC for binary targets
(MAPIE 1.x rejects APS/RAPS at runtime with a `ValueError: Invalid conformity
score for binary target. The only valid score is 'lac'`). The interesting
axis to vary is therefore the **splitting strategy**, not the score:

* **Split CP (LAC, prefit XGB)** — the production model. Calibrates on a
  single 20% cal split. Cheap; one model trained.
* **Cross-CP, k=5 (LAC, Jackknife+ flavour)** — `CrossConformalClassifier`
  with cv=5. Uses every training row for both fitting and calibration via
  K-fold; trains 5 models. More data-efficient on small samples (n=303
  here) at the cost of K-fold compute.
* **Cross-CP, k=10 (LAC)** — same as above with cv=10. Closer to
  leave-one-out; smaller per-fold variance, more compute.

For every (method, alpha) pair we report empirical coverage and mean set
size on the test split. The grouped bar chart visualises the trade-off:
all three methods clear the 1-alpha guarantee, and the smaller bar wins.
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
from mapie.classification import CrossConformalClassifier, SplitConformalClassifier
from xgboost import XGBClassifier

from src.logger import get_logger

logger = get_logger(__name__)

MethodResults = Dict[str, Dict[str, Dict[str, float]]]


def _xgb_from_cfg(xgb_cfg: dict[str, Any], seed: int) -> XGBClassifier:
    """Construct an XGBClassifier from the project config block."""
    cfg = dict(xgb_cfg)
    cfg.pop("use_label_encoder", None)  # removed in XGBoost 2.x
    cfg.pop("random_state", None)
    cfg.setdefault("eval_metric", "logloss")
    return XGBClassifier(
        n_estimators=int(cfg.pop("n_estimators", 200)),
        max_depth=int(cfg.pop("max_depth", 4)),
        learning_rate=float(cfg.pop("learning_rate", 0.05)),
        random_state=seed,
        **cfg,
    )


def _score_split(
    scc: SplitConformalClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
) -> dict[str, dict[str, float]]:
    """Score a fitted SplitConformalClassifier on (X_test, y_test) for each alpha."""
    metrics: dict[str, dict[str, float]] = {}
    _, y_set_all = scc.predict_set(X_test)
    n = int(len(y_test))
    for idx, alpha in enumerate(alphas):
        y_set = y_set_all[:, :, idx]
        covered = float(y_set[np.arange(n), y_test].sum() / n)
        sizes = y_set.sum(axis=1)
        metrics[str(alpha)] = {
            "empirical_coverage": covered,
            "mean_set_size": float(sizes.mean()),
        }
    return metrics


def _score_cv_plus(
    estimator: XGBClassifier,
    X_full: np.ndarray,
    y_full: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
    cv: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Fit a CrossConformalClassifier (CV+) at the given cv depth and score it."""
    confidence_levels = [1.0 - a for a in alphas]
    ccc = CrossConformalClassifier(
        estimator=estimator,
        confidence_level=confidence_levels,
        conformity_score="lac",
        cv=cv,
        random_state=seed,
    )
    ccc.fit_conformalize(X_full, y_full)
    _, y_set_all = ccc.predict_set(X_test)
    n = int(len(y_test))
    metrics: dict[str, dict[str, float]] = {}
    for idx, alpha in enumerate(alphas):
        y_set = y_set_all[:, :, idx]
        covered = float(y_set[np.arange(n), y_test].sum() / n)
        sizes = y_set.sum(axis=1)
        metrics[str(alpha)] = {
            "empirical_coverage": covered,
            "mean_set_size": float(sizes.mean()),
        }
    return metrics


def compare_methods(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
    xgb_params: dict[str, Any],
    seed: int,
) -> MethodResults:
    """Run Split-CP and two CV+ depths on the same test split.

    Returns a nested dict keyed by ``split_lac`` / ``cv_plus_5`` /
    ``cv_plus_10``; each entry maps alpha to {empirical_coverage,
    mean_set_size}. Mirrors the shape of
    ``coverage.compute_per_alpha_coverage`` so JSON readers handle either
    output uniformly.
    """
    confidence_levels = [1.0 - a for a in alphas]
    results: MethodResults = {}

    # Split-CP LAC — fit XGB on train, conformalize on cal.
    xgb_split = _xgb_from_cfg(xgb_params, seed).fit(X_train, y_train)
    scc_split = SplitConformalClassifier(
        estimator=xgb_split,
        confidence_level=confidence_levels,
        conformity_score="lac",
        prefit=True,
    )
    scc_split.conformalize(X_cal, y_cal)
    results["split_lac"] = _score_split(scc_split, X_test, y_test, alphas)
    logger.info("Compared method=split_lac alphas=%s", alphas)

    # Cross-CP / Jackknife+ flavour — fit on train+cal, no prefit.
    X_full = np.vstack([X_train, X_cal])
    y_full = np.concatenate([y_train, y_cal])
    for k, key in ((5, "cv_plus_5"), (10, "cv_plus_10")):
        xgb_cv = _xgb_from_cfg(xgb_params, seed)
        results[key] = _score_cv_plus(
            xgb_cv, X_full, y_full, X_test, y_test, alphas, cv=k, seed=seed
        )
        logger.info("Compared method=%s alphas=%s", key, alphas)

    return results


def plot_method_comparison(results: MethodResults, out_path: Path) -> None:
    """Render a 2-panel chart: empirical coverage and mean set size per method."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    methods = list(results.keys())
    alphas_sorted = sorted(next(iter(results.values())).keys(), key=float)

    fig, (ax_cov, ax_size) = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.25
    x = np.arange(len(alphas_sorted))
    palette = ["#4f46e5", "#7c3aed", "#0ea5e9"]
    for i, method in enumerate(methods):
        cov = [results[method][a]["empirical_coverage"] for a in alphas_sorted]
        size = [results[method][a]["mean_set_size"] for a in alphas_sorted]
        offset = (i - (len(methods) - 1) / 2) * width
        ax_cov.bar(x + offset, cov, width=width, label=method, color=palette[i % 3])
        ax_size.bar(x + offset, size, width=width, label=method, color=palette[i % 3])

    for ax_idx, ax in enumerate((ax_cov, ax_size)):
        ax.set_xticks(x)
        ax.set_xticklabels([f"alpha={a}" for a in alphas_sorted])
        ax.legend(loc="best")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        if ax_idx == 0:
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Empirical coverage")
            for j, a in enumerate(alphas_sorted):
                ax.hlines(
                    1.0 - float(a),
                    j - 0.45,
                    j + 0.45,
                    colors="#dc2626",
                    linestyles="dashed",
                )

    ax_cov.set_title("Coverage by method")
    ax_size.set_title("Mean set size by method")
    fig.suptitle("Method comparison: Split CP vs CV+ (k=5, 10) on test split")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote method-comparison chart to %s", out_path)
