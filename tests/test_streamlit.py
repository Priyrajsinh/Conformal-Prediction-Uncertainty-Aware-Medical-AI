"""Smoke tests for the Streamlit dashboard (Day 6).

Covers:
  - app module imports without error
  - glassmorphism CSS file is present
  - hero text renders via AppTest
  - all four tab labels are present via AppTest
"""

import importlib
from pathlib import Path

import pytest


def test_app_module_imports() -> None:
    """app.py at project root must be importable without raising."""
    importlib.import_module("app")


def test_glass_css_exists() -> None:
    """src/api/streamlit_glass.css must exist (rule C14/C44)."""
    assert Path("src/api/streamlit_glass.css").exists()


@pytest.fixture(scope="module")
def _app_test():
    """Return a run AppTest instance for the root app.py."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=60)
    at.run()
    return at


def test_hero_renders(_app_test) -> None:
    """Hero section must contain the project title text."""
    combined = " ".join(b.value for b in _app_test.markdown)
    assert (
        "Conformal Prediction" in combined
    ), "Hero heading not found in rendered markdown blocks"


def test_four_tabs_present(_app_test) -> None:
    """Dashboard must expose exactly 4 tabs."""
    assert len(_app_test.tabs) == 4, f"Expected 4 tabs, found {len(_app_test.tabs)}"
