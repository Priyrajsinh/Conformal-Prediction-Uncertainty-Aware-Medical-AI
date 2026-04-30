"""Generate EDA figures from the UCI Heart Disease dataset."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

plt.switch_backend("Agg")

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


def main() -> None:
    """Load data, clean, and save three EDA figures."""
    from pathlib import Path

    FIGURES_DIR = Path("reports/figures/eda")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("data/raw/heart.csv", header=None, names=COLUMNS, na_values=["?"])
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Dropped {n_before - len(df)} rows; {len(df)} remain")

    for col in COLUMNS:
        if col == "oldpeak":
            df[col] = df[col].astype(float)
        else:
            df[col] = df[col].astype(int)

    df["cp"] = df["cp"].map({1: 0, 2: 1, 3: 2, 4: 3})
    df["slope"] = df["slope"].map({1: 0, 2: 1, 3: 2})
    df["thal"] = df["thal"].map({3: 0, 6: 1, 7: 2})
    df["target"] = (df["target"] > 0).astype(int)

    print(f"Target:\n{df['target'].value_counts().to_string()}\n")

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df["target"].value_counts().sort_index()
    ax.bar(
        ["No Disease (0)", "Disease (1)"], counts.values, color=["steelblue", "tomato"]
    )
    ax.set_title("Target Distribution")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)
    print("Saved target_distribution.png")

    # 2. Feature distributions by target
    features = [c for c in COLUMNS if c != "target"]
    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    axes_flat = axes.flatten()
    for i, feat in enumerate(features):
        for label, color in [(0, "steelblue"), (1, "tomato")]:
            axes_flat[i].hist(
                df.loc[df["target"] == label, feat],
                bins=20,
                alpha=0.6,
                color=color,
                label=str(label),
            )
        axes_flat[i].set_title(feat)
        axes_flat[i].legend(title="target")
    for j in range(len(features), len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Feature Distributions by Target Class", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_distributions.png", dpi=150)
    plt.close(fig)
    print("Saved feature_distributions.png")

    # 3. Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Feature Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved correlation_heatmap.png")


if __name__ == "__main__":
    main()
