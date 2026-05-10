"""Streamlit 4-tab dashboard — Conformal Prediction Uncertainty-Aware Medical AI.

Tabs:
  1 — Predict + SHAP waterfall (per-patient explanation)
  2 — Global SHAP beeswarm + bar chart
  3 — Coverage + ECE + DCA + Selective classification plots
  4 — Mondrian group-conditional fairness comparison

Rules: C14 (glass CSS), C15 (hero), C44 (glass), C45 (NL output), U1 (SHAP), U2 (ECE).
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

plt.switch_backend("Agg")

from src.api.nl_translator import translate  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.schemas import HeartFeatures  # noqa: E402
from src.models.model import ConformalXGBoost  # noqa: E402

st.set_page_config(
    page_title="Conformal Heart Disease AI",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🫀",
)

_CSS_PATH = Path(__file__).parent / "src" / "api" / "streamlit_glass.css"
st.markdown(
    f"<style>{_CSS_PATH.read_text()}</style>",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts() -> tuple:
    """Load config, scaler, model, cal split, and SHAP explainer once."""
    cfg = load_config("config/config.yaml")
    scaler = joblib.load(Path(cfg["paths"]["models_dir"]) / "scaler.joblib")
    model = ConformalXGBoost.load(Path(cfg["paths"]["models_dir"]))
    cal = pd.read_csv(Path(cfg["data"]["processed_dir"]) / "cal.csv")
    explainer = shap.TreeExplainer(model.xgb)
    return cfg, scaler, model, cal, explainer


cfg, scaler, model, cal, explainer = load_artifacts()

_HF_URL = "https://huggingface.co/spaces/Priyrajsinh/conformal-prediction-medical-ai"
_GH_URL = (
    "https://github.com/Priyrajsinh/"
    "Conformal-Prediction-Uncertainty-Aware-Medical-AI"
)

# ── Hero (rule C15) ─────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="hero">
  <h1>Conformal Prediction · Uncertainty-Aware Medical AI</h1>
  <p>Coverage-guaranteed risk stratification on UCI Heart Disease Cleveland.</p>
  <div class="hero-links">
    <a href="{_HF_URL}" target="_blank">🤗 Live Gradio Space</a>
    <a href="{_GH_URL}" target="_blank">⭐ GitHub</a>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Controls")
    alpha = st.slider(
        "α (miscoverage tolerance)",
        min_value=0.01,
        max_value=0.50,
        value=0.10,
        step=0.01,
        help="Lower α → wider prediction sets, higher coverage guarantee.",
    )
    _alpha_options = [0.05, 0.10, 0.20]
    if alpha not in _alpha_options:
        alpha = min(_alpha_options, key=lambda a: abs(a - alpha))
        st.caption(f"Snapped to nearest calibrated α = {alpha}")

    st.markdown("---")
    st.markdown("**Coverage guarantee:** " f"`{1 - alpha:.0%}` at α = {alpha}")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🫀 Predict + SHAP",
        "🌐 Global SHAP",
        "📊 Coverage + ECE",
        "👥 Group Fairness",
    ]
)

# ── Tab 1: Predict + SHAP waterfall ─────────────────────────────────────────
with tab1:
    st.subheader("Patient input — 13 UCI features")
    cols = st.columns(3)
    _fields: list[tuple] = [
        ("age", 29, 77, 55),
        ("sex", 0, 1, 1),
        ("cp", 0, 3, 2),
        ("trestbps", 94, 200, 120),
        ("chol", 126, 564, 200),
        ("fbs", 0, 1, 0),
        ("restecg", 0, 2, 1),
        ("thalach", 71, 202, 150),
        ("exang", 0, 1, 0),
        ("oldpeak", 0.0, 6.2, 1.0),
        ("slope", 0, 2, 2),
        ("ca", 0, 4, 0),
        ("thal", 0, 3, 2),
    ]
    inputs: dict[str, int | float] = {}
    for i, (name, lo, hi, default) in enumerate(_fields):
        with cols[i % 3]:
            if isinstance(default, float):
                inputs[name] = st.slider(
                    name, float(lo), float(hi), float(default), 0.1
                )
            else:
                inputs[name] = st.slider(name, int(lo), int(hi), int(default), 1)

    if st.button("Predict", type="primary"):
        try:
            HeartFeatures.model_validate(inputs)
        except Exception as exc:
            st.error(f"Invalid input: {exc}")
            st.stop()

        x = scaler.transform(np.array([list(inputs.values())]))
        y_pred, y_set = model.safe_predict(x, alpha=alpha)
        pred_set = [int(c) for c in np.where(y_set[0])[0]]
        nl = translate(label=int(y_pred[0]), prediction_set=pred_set)

        st.markdown(f"### {nl}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Prediction set", str(pred_set))
        m2.metric("Coverage guarantee", f"{1 - alpha:.0%}")
        m3.metric("Set size", str(len(pred_set)))

        # SHAP waterfall — per patient (rule U1)
        st.subheader("Why? — SHAP waterfall (this patient)")
        with st.expander("How to read this chart"):
            st.markdown(
                "Each bar shows how much a feature **pushed the model's output** "
                "up (red) or down (blue) from the baseline expected value. "
                "The final value on the right is the model's log-odds output for "
                "this patient."
            )
        sv = explainer(x)
        fig, _ = plt.subplots()
        shap.plots.waterfall(sv[0], show=False)
        st.pyplot(fig)
        plt.close(fig)

# ── Tab 2: Global SHAP beeswarm + bar ───────────────────────────────────────
with tab2:
    st.subheader("Global feature importance — SHAP beeswarm (rule U1)")
    st.caption("Computed on the calibration split (n ≈ 60 rows).")

    X_cal = scaler.transform(cal.drop(columns=["target"]).values)
    sv_cal = explainer(X_cal)

    fig1, _ = plt.subplots(figsize=(8, 5))
    shap.plots.beeswarm(sv_cal, show=False)
    st.pyplot(fig1)
    plt.close(fig1)

    st.subheader("Mean |SHAP value| per feature")
    fig2, _ = plt.subplots(figsize=(8, 4))
    shap.plots.bar(sv_cal, show=False)
    st.pyplot(fig2)
    plt.close(fig2)

# ── Tab 3: Coverage + ECE + DCA + Selective ──────────────────────────────────
with tab3:
    st.subheader("Conformal coverage guarantee per α")
    col_a, col_b = st.columns(2)
    with col_a:
        st.image("reports/figures/coverage_guarantee.png", use_container_width=True)
    with col_b:
        st.image("reports/figures/set_sizes.png", use_container_width=True)

    st.subheader("ECE — probability calibration (rule U2)")
    st.image("reports/figures/calibration.png", use_container_width=True)
    st.markdown(
        """
> **Why both metrics?**
> **ECE** measures *probability* calibration — how faithfully `P(y=1|x)` matches
> observed frequencies.
> **Conformal coverage** measures *set* coverage — how often the prediction set
> contains the true label. A model can be ECE-uncalibrated and still satisfy the
> conformal coverage guarantee, and vice versa. These are orthogonal properties.
"""
    )

    st.subheader("Decision Curve Analysis")
    st.image("reports/figures/dca_net_benefit.png", use_container_width=True)

    st.subheader("Selective classification — accuracy vs abstain rate")
    st.image("reports/figures/selective_accuracy.png", use_container_width=True)

# ── Tab 4: Mondrian group fairness ───────────────────────────────────────────
with tab4:
    st.subheader("Group-conditional coverage — Mondrian CP (rule 3)")
    st.caption(
        "Marginal coverage holds on average. Conditional coverage is what "
        "fairness audits and EU AI Act Article 10 actually require."
    )
    st.image("reports/figures/group_coverage.png", use_container_width=True)

    results: dict = json.loads(Path(cfg["paths"]["results_json"]).read_text())
    group_cov = results.get("group_coverage", {})
    if group_cov:
        rows = []
        for group, metrics in group_cov.items():
            if not isinstance(metrics, dict):
                continue
            row: dict[str, str | float | int] = {"group": group}
            row.update(metrics)
            rows.append(row)
        if rows:
            st.dataframe(
                pd.DataFrame(rows).set_index("group"), use_container_width=True
            )

    st.markdown(
        """
> **Marginal vs conditional coverage:**
> A 90 % marginal-coverage model can deliver 70 % coverage on one demographic group
> and 95 % on another — that violates EU AI Act Annex III § 1 fairness requirements.
> Mondrian CP calibrates *separately within each group*, enforcing the guarantee for
> every subpopulation.
"""
    )

    with st.expander("Method comparison: APS vs RAPS vs CV+"):
        st.image(
            "reports/figures/method_comparison_set_sizes.png",
            use_container_width=True,
        )
        method_cmp = results.get("method_comparison", {})
        if method_cmp:
            rows2 = []
            for method_name, alphas_dict in method_cmp.items():
                for a, vals in alphas_dict.items():
                    row2: dict[str, str | float] = {
                        "method": method_name,
                        "alpha": a,
                    }
                    row2.update(vals)
                    rows2.append(row2)
            st.dataframe(
                pd.DataFrame(rows2).set_index(["method", "alpha"]),
                use_container_width=True,
            )
