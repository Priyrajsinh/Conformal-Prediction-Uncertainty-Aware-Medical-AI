"""Tests for src/data/preprocessing.py — three-way split, scaler, save."""

import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import three_way_split


def _make_df(n: int = 100) -> pd.DataFrame:
    """Return a synthetic DataFrame shaped like the heart disease dataset."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "age": rng.integers(29, 78, n),
            "sex": rng.integers(0, 2, n),
            "cp": rng.integers(0, 4, n),
            "trestbps": rng.integers(94, 201, n),
            "chol": rng.integers(126, 565, n),
            "fbs": rng.integers(0, 2, n),
            "restecg": rng.integers(0, 3, n),
            "thalach": rng.integers(71, 203, n),
            "exang": rng.integers(0, 2, n),
            "oldpeak": rng.uniform(0.0, 6.2, n).astype(float),
            "slope": rng.integers(0, 3, n),
            "ca": rng.integers(0, 5, n),
            "thal": rng.integers(0, 4, n),
            "target": rng.integers(0, 2, n),
        }
    )
    return df


class TestThreeWaySplit:
    """Tests for three_way_split."""

    def test_sizes_sum_to_input(self) -> None:
        """len(train) + len(cal) + len(test) equals original df length."""
        df = _make_df(200)
        train, cal, test = three_way_split(df, 0.6, 0.2, 0.2, "target", 42)
        assert len(train) + len(cal) + len(test) == len(df)

    def test_zero_overlap_train_cal(self) -> None:
        """train and cal share no rows."""
        df = _make_df(200)
        train, cal, _ = three_way_split(df, 0.6, 0.2, 0.2, "target", 42)
        train_set = set(map(tuple, train.values.tolist()))
        cal_set = set(map(tuple, cal.values.tolist()))
        assert train_set.isdisjoint(cal_set)

    def test_zero_overlap_cal_test(self) -> None:
        """cal and test share no rows."""
        df = _make_df(200)
        _, cal, test = three_way_split(df, 0.6, 0.2, 0.2, "target", 42)
        cal_set = set(map(tuple, cal.values.tolist()))
        test_set = set(map(tuple, test.values.tolist()))
        assert cal_set.isdisjoint(test_set)

    def test_zero_overlap_train_test(self) -> None:
        """train and test share no rows."""
        df = _make_df(200)
        train, _, test = three_way_split(df, 0.6, 0.2, 0.2, "target", 42)
        train_set = set(map(tuple, train.values.tolist()))
        test_set = set(map(tuple, test.values.tolist()))
        assert train_set.isdisjoint(test_set)

    def test_stratification_preserves_ratio(self) -> None:
        """Target class ratio in each split is within 10% of the full dataset."""
        df = _make_df(300)
        overall_rate = df["target"].mean()
        train, cal, test = three_way_split(df, 0.6, 0.2, 0.2, "target", 42)
        for name, split in [("train", train), ("cal", cal), ("test", test)]:
            rate = split["target"].mean()
            assert (
                abs(rate - overall_rate) < 0.10
            ), f"{name} target rate {rate:.3f} diverges from {overall_rate:.3f}"

    def test_invalid_ratios_raise(self) -> None:
        """AssertionError raised when split percentages do not sum to 1.0."""
        df = _make_df(100)
        with pytest.raises(AssertionError):
            three_way_split(df, 0.5, 0.3, 0.3, "target", 42)
