"""Day 2 data preparation: load → split → scale → baseline → save artefacts."""

from pathlib import Path

import yaml

from src.baseline.baseline import train_baseline
from src.data.dataset import load_heart_disease
from src.data.preprocessing import (
    FEATURE_COLS,
    fit_scaler,
    save_splits,
    three_way_split,
)


def main() -> None:
    """Run the full Day 2 pipeline: split data, fit scaler, train baseline."""
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    raw_path = Path(cfg["data"]["raw_path"])
    processed_dir = Path(cfg["data"]["processed_dir"])
    models_dir = Path(cfg["paths"]["models_dir"])
    figures_dir = Path(cfg["paths"]["figures_dir"])
    results_path = Path(cfg["paths"]["reports_dir"]) / "results.json"

    df = load_heart_disease(raw_path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns.")
    print(f"Target distribution:\n{df['target'].value_counts().to_string()}\n")

    train, cal, test = three_way_split(
        df,
        cfg["data"]["train_pct"],
        cfg["data"]["cal_pct"],
        cfg["data"]["test_pct"],
        stratify_col="target",
        seed=cfg["data"]["random_seed"],
    )
    print(f"Splits — train: {len(train)}  cal: {len(cal)}  test: {len(test)}")

    train_s, cal_s, test_s = fit_scaler(
        train,
        cal,
        test,
        FEATURE_COLS,
        models_dir / "scaler.joblib",
    )

    save_splits(train_s, cal_s, test_s, processed_dir)
    print("Processed splits saved.")

    xgb_params: dict[str, int | float | str | bool] = {
        k: v for k, v in cfg["model"]["xgb"].items() if k not in ("use_label_encoder",)
    }
    train_baseline(
        train_s,
        test_s,
        figures_dir,
        results_path,
        cfg["training"]["mlflow_experiment"],
        xgb_params,
    )
    print("Baseline training complete. Reports saved.")


if __name__ == "__main__":
    main()
