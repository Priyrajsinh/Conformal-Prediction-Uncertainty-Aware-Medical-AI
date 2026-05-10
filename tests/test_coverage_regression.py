"""Coverage regression test (rule B / CLAUDE.md C37).

Loads reports/results.json and asserts that empirical coverage >= 1 - alpha
for every calibrated alpha. Runs in CI only when results.json is present.
"""

import json
from pathlib import Path

import pytest

_RESULTS = Path("reports/results.json")
_ALPHAS = [0.05, 0.10, 0.20]


@pytest.fixture(scope="module")
def coverage_results() -> dict:
    assert _RESULTS.exists(), f"{_RESULTS} not found — run `make train` first"
    return json.loads(_RESULTS.read_text()).get("coverage", {})


@pytest.mark.parametrize("alpha", _ALPHAS)
def test_empirical_coverage_meets_guarantee(
    coverage_results: dict, alpha: float
) -> None:
    """Empirical coverage on the test split must be >= 1 - alpha."""
    key = str(alpha).rstrip("0").rstrip(".")
    if key not in coverage_results:
        key = str(alpha)
    assert (
        key in coverage_results
    ), f"No coverage entry for alpha={alpha} in results.json"
    empirical = coverage_results[key]["empirical_coverage"]
    guarantee = 1.0 - alpha
    assert empirical >= guarantee, (
        f"Coverage violation at alpha={alpha}: "
        f"empirical={empirical:.4f} < guarantee={guarantee:.4f}"
    )
