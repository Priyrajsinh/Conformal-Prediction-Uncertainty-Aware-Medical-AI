"""Coverage regression test (rule B / CLAUDE.md C37).

Loads reports/results.json (committed, always present) and asserts:
  1. Empirical coverage >= 1 - alpha - TOLERANCE for every calibrated alpha.
  2. ECE < ECE_CAP (sanity check on calibration quality).
  3. Group-conditional coverage deviates < GAP_CAP from marginal (n >= 20 only).

NEVER lower TOLERANCE or ECE_CAP to make a failing test pass. A failure here
means the conformal guarantee is violated — it is a real bug.
"""

import json
from pathlib import Path

import pytest

_RESULTS = Path("reports/results.json")
TOLERANCE = 0.02  # 2pp grace below the theoretical 1-alpha guarantee
ECE_CAP = 0.15  # sanity threshold; ECE > 0.15 indicates poor calibration
GAP_CAP = 0.05  # max allowed deviation of subgroup coverage from marginal
MIN_GROUP_N = 20  # skip groups too small for reliable coverage estimates


@pytest.fixture(scope="module")
def results() -> dict:
    """Load full evaluation results; skip gracefully when file is absent."""
    if not _RESULTS.exists():
        pytest.skip("reports/results.json not yet generated — run evaluate.py first")
    return json.loads(_RESULTS.read_text())


def test_coverage_meets_guarantee_for_every_alpha(results: dict) -> None:
    """Empirical coverage on the test split must be >= 1 - alpha - TOLERANCE."""
    coverage = results["coverage"]
    for alpha_str, m in coverage.items():
        alpha = float(alpha_str)
        target = (1.0 - alpha) - TOLERANCE
        assert m["empirical_coverage"] >= target, (
            f"Coverage violation at alpha={alpha}: "
            f"empirical={m['empirical_coverage']:.4f} < target={target:.4f}"
        )


def test_ece_below_sanity_threshold(results: dict) -> None:
    """Expected Calibration Error must stay below the sanity cap."""
    ece = results["ece"]["value"]
    assert ece < ECE_CAP, (
        f"ECE {ece:.4f} exceeds sanity cap {ECE_CAP}. "
        "Model probability calibration has degraded."
    )


def test_no_subgroup_underperforms_by_more_than_gap_cap(results: dict) -> None:
    """Group-conditional coverage at alpha=0.10 must be within GAP_CAP of marginal.

    Groups with fewer than MIN_GROUP_N samples are skipped — coverage estimates
    for small groups have wide confidence intervals and are not reliable.
    """
    if "group_coverage" not in results:
        pytest.skip("group_coverage not present in results.json")

    marginal = results["coverage"]["0.1"]["empirical_coverage"]
    group_cov = results["group_coverage"]

    for grp, m in group_cov.items():
        if not isinstance(m, dict):
            continue
        if "empirical_coverage" not in m:
            continue
        n = m.get("n", 0)
        if n < MIN_GROUP_N:
            continue
        gap = abs(marginal - m["empirical_coverage"])
        assert gap < GAP_CAP, (
            f"Subgroup '{grp}' (n={n}) coverage {m['empirical_coverage']:.4f} "
            f"deviates from marginal {marginal:.4f} by {gap:.4f} >= {GAP_CAP}"
        )
