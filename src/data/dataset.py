"""Load, clean, remap, and validate the UCI Heart Disease (Cleveland) dataset."""

import hashlib
from pathlib import Path

import pandas as pd

from src.data.validation import validate_heart_df
from src.exceptions import ChecksumError, DataLoadError
from src.logger import get_logger

logger = get_logger(__name__)

COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# Raw Cleveland codes → 0-indexed values (aligns with CLAUDE.md API schema).
_CP_MAP = {1: 0, 2: 1, 3: 2, 4: 3}
_SLOPE_MAP = {1: 0, 2: 1, 3: 2}
_THAL_MAP = {3: 0, 6: 1, 7: 2}


def verify_checksum(path: Path) -> None:
    """Raise ChecksumError if SHA-256 of *path* does not match the .sha256 sidecar."""
    sidecar = path.parent / f"{path.name}.sha256"
    expected = sidecar.read_text().strip()
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != actual:
        raise ChecksumError(
            f"SHA-256 mismatch for {path}: "
            f"expected {expected[:8]}..., got {actual[:8]}..."
        )


def load_heart_disease(path: str | Path) -> pd.DataFrame:
    """Load, clean, remap, and validate the UCI Heart Disease Cleveland CSV.

    Steps: read CSV → drop NaN rows → cast dtypes → remap categorical codes
    to 0-indexed → binarise target → pandera validation.
    """
    path = Path(path)
    if not path.exists():
        raise DataLoadError(f"Heart CSV not found at {path}")
    verify_checksum(path)

    df = pd.read_csv(path, header=None, names=COLUMNS, na_values=["?"])
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    logger.info("Dropped %d rows with NaN; kept %d", n_before - len(df), len(df))

    for col in COLUMNS:
        if col == "oldpeak":
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(int)

    # Remap Cleveland-coded categoricals to 0-indexed (cp: 1-4→0-3, etc.)
    df["cp"] = df["cp"].map(_CP_MAP)
    df["slope"] = df["slope"].map(_SLOPE_MAP)
    df["thal"] = df["thal"].map(_THAL_MAP)

    df["target"] = (df["target"] > 0).astype(int)
    return validate_heart_df(df)
