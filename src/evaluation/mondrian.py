"""Group-conditional (Mondrian) coverage and parity statistical tests.

Marginal conformal coverage is just an average — a model can clear the
1-alpha guarantee overall while systematically under-covering a subgroup
(women, elderly, etc.). Mondrian conformal prediction reports per-subgroup
empirical coverage so that imbalance shows up as a number rather than a
hidden failure mode.

Two parity tests are reported on top of the per-group numbers:

* **Sex parity** — single chi-squared test on the 2x2 contingency table
  (covered vs not, sex==0 vs sex==1). p < 0.05 flags coverage that is
  meaningfully different between sexes.
* **Age parity (Bonferroni-corrected)** — three age-bin chi-squared tests
  versus the rest of the cohort (one-vs-rest), p-values multiplied by 3
  to control family-wise error.

The plot ``group_coverage.png`` shows each subgroup's empirical coverage
versus the 1 - alpha reference line.
"""

from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from src.logger import get_logger
from src.models.model import ConformalXGBoost

logger = get_logger(__name__)

GroupResults = Dict[str, Any]


def _build_masks(raw_test: pd.DataFrame) -> dict[str, np.ndarray]:
    """Return boolean masks for the five reportable subgroups."""
    age = raw_test["age"].astype(int).values
    sex = raw_test["sex"].astype(int).values
    return {
        "sex_0": sex == 0,
        "sex_1": sex == 1,
        "age_lt_50": age < 50,
        "age_50_64": (age >= 50) & (age < 65),
        "age_65p": age >= 65,
    }


def group_coverage(
    model: ConformalXGBoost,
    raw_test: pd.DataFrame,
    X_test: np.ndarray,
    alpha: float,
) -> GroupResults:
    """Compute empirical coverage per subgroup at the given alpha.

    ``raw_test`` carries the unscaled sex/age columns used to build the
    Mondrian masks; ``X_test`` is the already-scaled feature matrix that
    feeds the model. Keeping these separate avoids inverse-transforming
    the StandardScaler to recover integer demographic columns.
    """
    _, y_set = model.safe_predict(X_test, alpha=float(alpha))
    y_true = raw_test["target"].astype(int).values
    masks = _build_masks(raw_test)
    results: GroupResults = {}
    for name, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            results[name] = {
                "n": 0,
                "empirical_coverage": float("nan"),
                "mean_set_size": float("nan"),
            }
            continue
        sub_set = y_set[mask]
        sub_y = y_true[mask]
        covered = float(sub_set[np.arange(n), sub_y].sum() / n)
        sizes = sub_set.sum(axis=1)
        results[name] = {
            "n": n,
            "empirical_coverage": covered,
            "mean_set_size": float(sizes.mean()),
        }
        logger.info(
            "group=%s n=%d coverage=%.3f mean_size=%.2f",
            name,
            n,
            covered,
            float(sizes.mean()),
        )
    results["overall_alpha"] = alpha
    return results


def _covered_indicator(
    model: ConformalXGBoost,
    raw_test: pd.DataFrame,
    X_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Return a binary array of whether each test row's true label was covered."""
    _, y_set = model.safe_predict(X_test, alpha=float(alpha))
    y_true = raw_test["target"].astype(int).values
    n = int(len(y_true))
    return np.asarray(y_set[np.arange(n), y_true]).astype(int)


def parity_test(
    model: ConformalXGBoost,
    raw_test: pd.DataFrame,
    X_test: np.ndarray,
    alpha: float,
) -> dict[str, float]:
    """Sex parity (one chi-squared) plus age parity (3 tests, Bonferroni).

    Returns a dict with ``sex_p_value``, ``age_lt_50_p_value_bonf``,
    ``age_50_64_p_value_bonf``, ``age_65p_p_value_bonf``. Bonferroni-corrected
    p-values are capped at 1.0; the family size is the number of age bins (3).
    """
    covered = _covered_indicator(model, raw_test, X_test, alpha)
    masks = _build_masks(raw_test)
    out: dict[str, float] = {}

    sex_table = np.array(
        [
            [
                int(covered[masks["sex_0"]].sum()),
                int((1 - covered[masks["sex_0"]]).sum()),
            ],
            [
                int(covered[masks["sex_1"]].sum()),
                int((1 - covered[masks["sex_1"]]).sum()),
            ],
        ]
    )
    if sex_table.sum() == 0 or (sex_table.sum(axis=1) == 0).any():
        out["sex_p_value"] = 1.0
    else:
        _, p, _, _ = chi2_contingency(sex_table)
        out["sex_p_value"] = float(p)

    age_keys = ("age_lt_50", "age_50_64", "age_65p")
    family = len(age_keys)
    for key in age_keys:
        in_mask = masks[key]
        out_mask = ~in_mask
        table = np.array(
            [
                [
                    int(covered[in_mask].sum()),
                    int((1 - covered[in_mask]).sum()),
                ],
                [
                    int(covered[out_mask].sum()),
                    int((1 - covered[out_mask]).sum()),
                ],
            ]
        )
        if table.sum() == 0 or (table.sum(axis=1) == 0).any():
            p_bonf = 1.0
        else:
            _, p, _, _ = chi2_contingency(table)
            p_bonf = min(1.0, float(p) * family)
        out[f"{key}_p_value_bonf"] = p_bonf
    return out


def plot_group_coverage(results: GroupResults, alpha: float, out_path: Path) -> None:
    """Render a bar chart of subgroup empirical coverage versus 1 - alpha."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    groups = [k for k in results.keys() if k != "overall_alpha" and k != "parity"]
    cov = [
        results[g]["empirical_coverage"]
        for g in groups
        if not np.isnan(results[g].get("empirical_coverage", np.nan))
    ]
    groups = [
        g for g in groups if not np.isnan(results[g].get("empirical_coverage", np.nan))
    ]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(groups))
    ax.bar(x, cov, width=0.55, color="#7c3aed")
    ax.hlines(
        1.0 - float(alpha),
        -0.5,
        len(groups) - 0.5,
        colors="#dc2626",
        linestyles="dashed",
        linewidth=2,
        label=f"target 1 - alpha = {1 - float(alpha):.2f}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_title(f"Mondrian group coverage at alpha = {alpha}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    logger.info("Wrote group-coverage chart to %s", out_path)
