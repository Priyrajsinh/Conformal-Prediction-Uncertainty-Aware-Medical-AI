"""Tests for src/data/dataset.py — loading, checksum, cleaning, binarisation."""

import hashlib
import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.data.dataset import load_heart_disease, verify_checksum
from src.exceptions import ChecksumError, DataLoadError

RAW_PATH = Path("data/raw/heart.csv")


class TestLoadHeartDisease:
    """Tests for load_heart_disease."""

    def test_returns_dataframe(self) -> None:
        """Loaded result is a non-empty DataFrame."""
        df = load_heart_disease(RAW_PATH)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_target_binarised(self) -> None:
        """target column contains only 0 and 1 after binarisation."""
        df = load_heart_disease(RAW_PATH)
        assert set(df["target"].unique()).issubset({0, 1})

    def test_no_nan_in_result(self) -> None:
        """Returned DataFrame has no NaN values."""
        df = load_heart_disease(RAW_PATH)
        assert not df.isnull().any().any()

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """DataLoadError raised when CSV does not exist."""
        with pytest.raises(DataLoadError):
            load_heart_disease(tmp_path / "nonexistent.csv")

    def test_handles_question_marks(self, tmp_path: Path) -> None:
        """Rows containing '?' are dropped; remaining rows are returned."""
        # Minimal valid Cleveland-format rows (two good, one with ?)
        csv_content = textwrap.dedent(
            """\
            63,1,1,145,233,1,2,150,0,2.3,3,0,3,0
            67,1,4,160,286,0,2,108,1,1.5,2,3,3,2
            41,0,?,130,204,0,0,172,0,1.4,1,0,3,0
            """
        )
        csv_file = tmp_path / "heart_test.csv"
        csv_file.write_text(csv_content)
        digest = hashlib.sha256(csv_file.read_bytes()).hexdigest()
        (tmp_path / "heart_test.csv.sha256").write_text(digest)

        df = load_heart_disease(csv_file)
        # Third row dropped; 2 rows remain
        assert len(df) == 2


class TestVerifyChecksum:
    """Tests for verify_checksum."""

    def test_valid_checksum_passes(self, tmp_path: Path) -> None:
        """No exception when checksum matches sidecar."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello")
        sidecar = tmp_path / "data.bin.sha256"
        sidecar.write_text(hashlib.sha256(b"hello").hexdigest())
        verify_checksum(f)  # should not raise

    def test_mismatch_raises(self, tmp_path: Path) -> None:
        """ChecksumError raised when sidecar has wrong hash."""
        f = tmp_path / "data.bin"
        f.write_bytes(b"hello")
        sidecar = tmp_path / "data.bin.sha256"
        sidecar.write_text("deadbeef" * 8)  # wrong hash
        with pytest.raises(ChecksumError):
            verify_checksum(f)
