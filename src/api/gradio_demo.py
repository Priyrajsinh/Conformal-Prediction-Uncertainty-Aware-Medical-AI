"""Streaming Gradio demo for conformal heart-disease prediction.

Run with:  python -m src.api.gradio_demo
Or:        make gradio
Then open: http://localhost:7860
Rules applied: C9, C14, C15, C43, C44.
"""

from pathlib import Path
from typing import Generator

import gradio as gr
import joblib
import matplotlib
import numpy as np

from src.api.nl_translator import translate
from src.api.theme import get_css, get_theme
from src.config import load_config
from src.data.schemas import HeartFeatures
from src.models.model import ConformalXGBoost

matplotlib.use("Agg")  # rule C15 -- non-interactive backend before any pyplot call

_CFG = load_config("config/config.yaml")
_MODELS_DIR = Path(_CFG["paths"]["models_dir"])

_scaler = joblib.load(_MODELS_DIR / "scaler.joblib")
_model: ConformalXGBoost = ConformalXGBoost.load(_MODELS_DIR)

_GITHUB_URL = (
    "https://github.com/Priyrajsinh/"
    "Conformal-Prediction-Uncertainty-Aware-Medical-AI"
)


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
    """Yield pipeline stage updates then the final conformal result.

    Stages (rule C43):
    1. Validating input
    2. Running XGBoost
    3. Constructing prediction set
    4. Translating to plain English
    5. Done (with full result payload)
    """
    yield "**Stage 1/4** -- Validating input...", None, ""

    features = HeartFeatures(
        age=age,
        sex=sex,
        cp=cp,
        trestbps=trestbps,
        chol=chol,
        fbs=fbs,
        restecg=restecg,
        thalach=thalach,
        exang=exang,
        oldpeak=oldpeak,
        slope=slope,
        ca=ca,
        thal=thal,
    )

    yield "**Stage 2/4** -- Running XGBoost classifier...", None, ""

    X = _scaler.transform(np.array([list(features.model_dump().values())]))

    yield (
        f"**Stage 3/4** -- Constructing prediction set (alpha={alpha:.2f})...",
        None,
        "",
    )

    y_pred, y_ps = _model.safe_predict(X, alpha=alpha)
    ps = y_ps.squeeze(-1) if y_ps.ndim == 3 else y_ps
    pred_set = [int(c) for c in np.where(ps[0])[0]]
    label = int(y_pred[0])
    mean_set_size = float(ps.sum(axis=1).mean())

    yield "**Stage 4/4** -- Translating to plain English...", None, ""

    nl = translate(label=label, prediction_set=pred_set)
    result = {
        "label": label,
        "prediction_set": pred_set,
        "coverage_guarantee": round(1 - alpha, 2),
        "mean_set_size": round(mean_set_size, 3),
        "alpha_used": alpha,
        "nl_summary": nl,
    }

    yield "**Done**", result, f"### Result\n{nl}"


with gr.Blocks(
    theme=get_theme(), css=get_css(), title="Conformal Heart Disease AI"
) as demo:
    gr.HTML(
        f"""
    <div class='hero'>
      <h1>Conformal Prediction - Uncertainty-Aware Medical AI</h1>
      <p>UCI Heart Disease - MAPIE LAC - Coverage guarantee 1 minus alpha</p>
      <a href='{_GITHUB_URL}' target='_blank'>GitHub</a>
    </div>
    """
    )

    with gr.Row():
        with gr.Column(scale=1):
            age_in = gr.Slider(29, 77, value=55, step=1, label="Age")
            sex_in = gr.Slider(0, 1, value=1, step=1, label="Sex (0=F, 1=M)")
            cp_in = gr.Slider(0, 3, value=2, step=1, label="Chest pain type (0-3)")
            trestbps_in = gr.Slider(
                94, 200, value=120, step=1, label="Resting BP (mmHg)"
            )
            chol_in = gr.Slider(
                126, 564, value=200, step=1, label="Cholesterol (mg/dl)"
            )
            fbs_in = gr.Slider(0, 1, value=0, step=1, label="Fasting blood sugar > 120")
            restecg_in = gr.Slider(0, 2, value=1, step=1, label="Resting ECG (0-2)")
            thalach_in = gr.Slider(71, 202, value=150, step=1, label="Max heart rate")
            exang_in = gr.Slider(0, 1, value=0, step=1, label="Exercise-induced angina")
            oldpeak_in = gr.Slider(
                0.0, 6.2, value=1.0, step=0.1, label="ST depression (oldpeak)"
            )
            slope_in = gr.Slider(0, 2, value=2, step=1, label="ST slope (0-2)")
            ca_in = gr.Slider(0, 4, value=0, step=1, label="Major vessels (0-4)")
            thal_in = gr.Slider(0, 3, value=2, step=1, label="Thal (0-3)")
            alpha_in = gr.Slider(
                0.01, 0.50, value=0.10, step=0.01, label="alpha (miscoverage)"
            )
            btn = gr.Button("Predict", variant="primary")

        with gr.Column(scale=2):
            stage_md = gr.Markdown(label="Pipeline stage")
            result_json = gr.JSON(label="Conformal output")
            nl_md = gr.Markdown(
                label="Clinical summary", elem_classes=["result-reveal"]
            )

    btn.click(
        fn=stream_classify,
        inputs=[
            age_in,
            sex_in,
            cp_in,
            trestbps_in,
            chol_in,
            fbs_in,
            restecg_in,
            thalach_in,
            exang_in,
            oldpeak_in,
            slope_in,
            ca_in,
            thal_in,
            alpha_in,
        ],
        outputs=[stage_md, result_json, nl_md],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)  # nosec B104
