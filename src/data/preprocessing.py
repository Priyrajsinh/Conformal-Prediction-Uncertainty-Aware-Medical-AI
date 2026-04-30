"""Three-way stratified split and StandardScaler fitting for the pipeline."""

import hashlib
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.logger import get_logger

logger = get_logger(__name__)

FEATURE_COLS = [
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
]


def three_way_split(
    df: pd.DataFrame,
    train_pct: float,
    cal_pct: float,
    test_pct: float,
    stratify_col: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, cal, test) via stratified splits with zero-overlap assertions.

    Split order: carve test from full df first, then carve cal from remainder.
    Asserting row-level disjointness satisfies rule C33.
    """
    assert abs(train_pct + cal_pct + test_pct - 1.0) < 1e-6, "splits must sum to 1.0"

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_pct, random_state=seed)
    rest_idx, test_idx = next(sss1.split(df, df[stratify_col]))
    rest = df.iloc[rest_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    cal_within_rest = cal_pct / (train_pct + cal_pct)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=cal_within_rest, random_state=seed
    )
    train_idx, cal_idx = next(sss2.split(rest, rest[stratify_col]))
    train = rest.iloc[train_idx].reset_index(drop=True)
    cal = rest.iloc[cal_idx].reset_index(drop=True)

    train_set = set(map(tuple, train.values.tolist()))
    cal_set = set(map(tuple, cal.values.tolist()))
    test_set = set(map(tuple, test.values.tolist()))
    assert train_set.isdisjoint(cal_set), "train ∩ cal not empty"
    assert cal_set.isdisjoint(test_set), "cal ∩ test not empty"
    assert train_set.isdisjoint(test_set), "train ∩ test not empty"

    logger.info(
        "Split sizes — train: %d  cal: %d  test: %d",
        len(train),
        len(cal),
        len(test),
    )
    return train, cal, test


def _write_checksum(path: Path) -> None:
    """Write SHA-256 sidecar .sha256 file alongside *path*."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.parent / f"{path.name}.sha256"
    sidecar.write_text(digest)


def fit_scaler(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    scaler_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit StandardScaler on train features only; transform cal and test.

    Scaler is fit on train to prevent data leakage (rule C35 analogue).
    """
    scaler = StandardScaler()
    train = train.copy()
    cal = cal.copy()
    test = test.copy()

    train[feature_cols] = scaler.fit_transform(train[feature_cols])
    cal[feature_cols] = scaler.transform(cal[feature_cols])
    test[feature_cols] = scaler.transform(test[feature_cols])

    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved → %s", scaler_path)
    return train, cal, test


def save_splits(
    train: pd.DataFrame,
    cal: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Save train/cal/test CSVs with SHA-256 sidecars to *out_dir*."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("cal", cal), ("test", test)]:
        path = out_dir / f"{name}.csv"
        split.to_csv(path, index=False)
        _write_checksum(path)
        logger.info("Saved %s (%d rows) → %s", name, len(split), path)
