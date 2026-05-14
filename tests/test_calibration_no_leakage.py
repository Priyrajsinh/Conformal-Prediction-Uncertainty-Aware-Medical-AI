"""Anti-leakage test: verify MAPIE was calibrated on the cal split only (rule C35).

models/calibration_metadata.json is gitignored (written by train.py). This test
skips cleanly when the file is absent (fresh clone without running train).
"""

import json
from pathlib import Path

import pytest

_META = Path("models/calibration_metadata.json")


@pytest.fixture(scope="module")
def meta() -> dict:
    """Load calibration metadata, skip if not generated yet."""
    if not _META.exists():
        pytest.skip("models/calibration_metadata.json not found — run train.py first")
    return json.loads(_META.read_text())


def test_calibration_used_cal_split_only(meta: dict) -> None:
    """MAPIE must be calibrated on the cal split, never on train or test."""
    assert (
        meta["split_used"] == "cal"
    ), f"Expected split_used='cal', got '{meta['split_used']}'"


def test_calibration_used_prefit_mode(meta: dict) -> None:
    """Calibration must use prefit=True so test split is never touched."""
    assert meta["prefit"] is True, "prefit must be True (rule C35)"


def test_calibration_used_lac_score_for_binary(meta: dict) -> None:
    """LAC conformity score required for binary targets in MAPIE 1.x (rule C35).

    RAPS adaptive ranking is degenerate on binary — only valid when n_classes >= 3.
    """
    got = meta["conformity_score"]
    assert got == "lac", f"Binary target requires conformity_score='lac', got '{got}'"


def test_calibration_set_large_enough(meta: dict) -> None:
    """Cal split must have enough samples for valid conformal guarantees."""
    assert meta["n_calibration_samples"] >= 50, (
        f"Cal split has only {meta['n_calibration_samples']} samples — "
        "too small for reliable CP guarantees"
    )
