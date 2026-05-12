"""Day 7 docs invariants — fail loudly if README/MODEL_CARD drift.

These tests are the regulatory contract for the documentation surface:
both live demo URLs must be present, no AI tool must be credited, no
``TBD`` placeholder must survive into MODEL_CARD, the five research
notes must exist, and the LICENSE file must be on disk.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

README = REPO_ROOT / "README.md"
MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"
LICENSE_FILE = REPO_ROOT / "LICENSE"
RESEARCH_NOTES_DIR = REPO_ROOT / "research-notes"

FORBIDDEN_AI_ATTRIBUTION = (
    "Claude",
    "Sonnet",
    "Opus",
    "Anthropic",
    "Co-Authored-By",
)

LIVE_URL_PATTERNS = (
    "huggingface.co/spaces/Priyrajsinh/conformal-prediction-medical-ai",
    "streamlit.app",
)


@pytest.fixture(scope="module")
def readme_text() -> str:
    """Return the contents of README.md."""
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def model_card_text() -> str:
    """Return the contents of MODEL_CARD.md."""
    return MODEL_CARD.read_text(encoding="utf-8")


def test_readme_has_live_urls(readme_text: str) -> None:
    """README must advertise both the HF Space and the Streamlit Cloud URLs."""
    for pattern in LIVE_URL_PATTERNS:
        assert pattern in readme_text, f"README missing live URL pattern: {pattern}"


def test_readme_no_ai_attribution(readme_text: str, model_card_text: str) -> None:
    """README and MODEL_CARD must not credit any AI assistant."""
    for token in FORBIDDEN_AI_ATTRIBUTION:
        assert token not in readme_text, f"README contains forbidden token: {token}"
        assert (
            token not in model_card_text
        ), f"MODEL_CARD contains forbidden token: {token}"


def test_model_card_has_no_TBD(model_card_text: str) -> None:
    """Every TBD must be replaced with a real number before shipping."""
    assert "TBD" not in model_card_text, "MODEL_CARD still contains a TBD placeholder"


def test_research_notes_has_5_md_files() -> None:
    """research-notes/ must hold exactly five numbered entries 01..05."""
    entries = sorted(RESEARCH_NOTES_DIR.glob("0?-*.md"))
    assert len(entries) == 5, f"expected 5 research-notes entries, got {len(entries)}"


def test_license_exists() -> None:
    """A top-level LICENSE file must be present (MIT)."""
    assert LICENSE_FILE.exists(), "LICENSE file is missing at repo root"
    assert "MIT License" in LICENSE_FILE.read_text(encoding="utf-8")
