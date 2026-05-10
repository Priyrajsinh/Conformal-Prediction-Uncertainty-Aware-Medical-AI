"""Self-contained Gradio app for the HuggingFace Space (rules C12, C13).

No imports from src/. Model artefacts (xgb.joblib, scc.joblib, meta.joblib,
scaler.joblib) are expected in the same directory as this file.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Generator

import matplotlib
import numpy as np

matplotlib.use("Agg")

import gradio as gr
import joblib
from mapie.classification import SplitConformalClassifier
from pydantic import BaseModel, Field
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic model (inlined — no src.* import)
# ---------------------------------------------------------------------------


class HeartFeatures(BaseModel):
    """13 UCI heart disease features with validated ranges."""

    age: int = Field(ge=29, le=77)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=94, le=200)
    chol: int = Field(ge=126, le=564)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=71, le=202)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0.0, le=6.2)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


# ---------------------------------------------------------------------------
# NL translator (inlined)
# ---------------------------------------------------------------------------


def translate(label: int, prediction_set: list[int]) -> str:
    """Convert a conformal prediction set to a patient-friendly sentence."""
    if len(prediction_set) != 1:
        return "Uncertain — recommend follow-up review by a clinician."
    return (
        "Likely heart disease (high confidence)."
        if label == 1
        else "Likely no heart disease (high confidence)."
    )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_scaler = joblib.load(_HERE / "scaler.joblib")

_xgb = joblib.load(_HERE / "xgb.joblib")
_scc: SplitConformalClassifier = joblib.load(_HERE / "scc.joblib")
_meta: dict = joblib.load(_HERE / "meta.joblib")
_alphas: list[float] = _meta.get("alphas", [0.05, 0.10, 0.20])

_logger.info("Model loaded: alphas=%s", _alphas)


def _safe_predict(
    X: np.ndarray, alpha: float
) -> tuple[np.ndarray, np.ndarray]:
    """Thin safe-predict wrapper that mirrors the src model's NaN/shape guards."""
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D X, got shape {X.shape}.")
    if np.isnan(X).any() or np.isinf(X).any():
        raise ValueError("Input contains NaN or inf.")
    if alpha not in _alphas:
        raise ValueError(f"alpha={alpha} not in calibrated alphas {_alphas}.")
    idx = _alphas.index(alpha)
    y_pred_arr, y_set_all = _scc.predict_set(X)
    y_set = y_set_all[:, :, idx]
    return np.asarray(y_pred_arr), np.asarray(y_set)


# ---------------------------------------------------------------------------
# Theme + CSS (inlined)
# ---------------------------------------------------------------------------


def _get_theme() -> gr.themes.Base:
    """Return indigo/purple Gradio theme."""
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
    )


def _get_css() -> str:
    """Return glassmorphism CSS."""
    return """
    body, .gradio-container {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
        min-height: 100vh;
    }
    .hero { text-align: center; padding: 2rem 1rem 1rem; animation: slideUp 0.6s ease-out; }
    .hero h1 {
        font-size: 2rem; font-weight: 800;
        background: linear-gradient(90deg, #a78bfa, #818cf8, #67e8f9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 0.5rem;
    }
    .hero p { color: #c4b5fd; font-size: 1rem; margin-bottom: 0.75rem; }
    .hero a { color: #818cf8; text-decoration: none; font-weight: 600; }
    .gr-box, .gr-form, .gr-panel, .block {
        background: rgba(255,255,255,0.07) !important;
        backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        border-radius: 16px !important;
    }
    button.primary {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important; font-weight: 700; transition: box-shadow 0.2s;
    }
    button.primary:hover { box-shadow: 0 0 24px rgba(139,92,246,0.7) !important; }
    .result-reveal { animation: slideUp 0.4s ease-out; }
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    label, .label-wrap span { color: #e0e7ff !important; font-weight: 600; }
    """


# ---------------------------------------------------------------------------
# Streaming predict function
# ---------------------------------------------------------------------------


def stream_classify(  # noqa: PLR0913
    age: int,
    sex: int,
    cp: int,
    trestbps: int,
    chol: int,
    fbs: int,
    restecg: int,
    thalach: int,
    exang: int,
    oldpeak: float,
    slope: int,
    ca: int,
    thal: int,
    alpha: float,
) -> Generator[tuple[str, dict | None, str], None, None]:
    """Yield 5 stage updates then the final ConformalOutput dict."""
    yield "**Stage 1/4** — Validating input...", None, ""

    features = HeartFeatures(
        age=age, sex=sex, cp=cp, trestbps=trestbps, chol=chol,
        fbs=fbs, restecg=restecg, thalach=thalach, exang=exang,
        oldpeak=oldpeak, slope=slope, ca=ca, thal=thal,
    )

    yield "**Stage 2/4** — Running XGBoost classifier...", None, ""

    X = _scaler.transform(np.array([list(features.model_dump().values())]))

    yield f"**Stage 3/4** — Constructing prediction set (α = {alpha:.2f})...", None, ""

    y_pred, y_ps = _safe_predict(X, alpha=alpha)
    ps = y_ps.squeeze(-1) if y_ps.ndim == 3 else y_ps
    pred_set = [int(c) for c in np.where(ps[0])[0]]
    label = int(y_pred[0])
    mean_set_size = float(ps.sum(axis=1).mean())

    yield "**Stage 4/4** — Translating to plain English...", None, ""

    nl = translate(label=label, prediction_set=pred_set)
    result = {
        "label": label,
        "prediction_set": pred_set,
        "coverage_guarantee": round(1.0 - alpha, 2),
        "mean_set_size": round(mean_set_size, 3),
        "alpha_used": alpha,
        "nl_summary": nl,
    }

    yield "**Done ✓**", result, f"### Result\n{nl}"


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Conformal Heart Disease AI", theme=_get_theme(), css=_get_css()) as demo:
    gr.HTML("""
    <div class='hero'>
      <h1>Conformal Prediction · Uncertainty-Aware Medical AI</h1>
      <p>UCI Heart Disease · MAPIE LAC · Coverage guarantee 1&minus;&alpha;</p>
      <a href='https://github.com/Priyrajsinh/Conformal-Prediction-Uncertainty-Aware-Medical-AI'
         target='_blank'>GitHub</a>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            _age = gr.Slider(29, 77, value=55, step=1, label="Age")
            _sex = gr.Slider(0, 1, value=1, step=1, label="Sex (0=F, 1=M)")
            _cp = gr.Slider(0, 3, value=2, step=1, label="Chest pain type (0-3)")
            _trestbps = gr.Slider(94, 200, value=120, step=1, label="Resting BP (mmHg)")
            _chol = gr.Slider(126, 564, value=200, step=1, label="Cholesterol (mg/dl)")
            _fbs = gr.Slider(0, 1, value=0, step=1, label="Fasting blood sugar > 120")
            _restecg = gr.Slider(0, 2, value=1, step=1, label="Resting ECG (0-2)")
            _thalach = gr.Slider(71, 202, value=150, step=1, label="Max heart rate")
            _exang = gr.Slider(0, 1, value=0, step=1, label="Exercise-induced angina")
            _oldpeak = gr.Slider(0.0, 6.2, value=1.0, step=0.1, label="ST depression")
            _slope = gr.Slider(0, 2, value=2, step=1, label="ST slope (0-2)")
            _ca = gr.Slider(0, 4, value=0, step=1, label="Major vessels (0-4)")
            _thal = gr.Slider(0, 3, value=2, step=1, label="Thal (0-3)")
            _alpha = gr.Slider(0.01, 0.50, value=0.10, step=0.01, label="α (miscoverage)")
            _btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=2):
            _stage = gr.Markdown(label="Pipeline stage")
            _result = gr.JSON(label="Conformal output")
            _nl = gr.Markdown(label="Clinical summary", elem_classes=["result-reveal"])

    _btn.click(
        fn=stream_classify,
        inputs=[
            _age, _sex, _cp, _trestbps, _chol, _fbs, _restecg,
            _thalach, _exang, _oldpeak, _slope, _ca, _thal, _alpha,
        ],
        outputs=[_stage, _result, _nl],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",  # nosec B104
        server_port=7860,
    )
