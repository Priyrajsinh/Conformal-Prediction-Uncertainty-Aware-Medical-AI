# Conformal Prediction · Uncertainty-Aware Medical AI

> Production-grade medical AI with mathematical coverage guarantees.
> UCI Heart Disease · MAPIE (RAPS, cv='prefit') · XGBoost · FastAPI · Gradio · Streamlit · MLflow.

**Status:** Day 6 complete — Streamlit 4-tab dashboard live. CI: 62 tests, 91.6% coverage.

## Live demos
- [🤗 Gradio Space](https://huggingface.co/spaces/Priyrajsinh/conformal-prediction-medical-ai) — streaming conformal pipeline with glassmorphism UI
- 🔗 Streamlit Cloud dashboard — *deploy in progress*
- 🔗 Swagger API docs (`make serve` → http://localhost:8000/docs)

## Quick start
```bash
make install         # pip install + pre-commit hooks
make train           # fit XGBoost + calibrate MAPIE on cal split
make test            # pytest + coverage gate (>=70%)
make serve           # FastAPI on :8000 (/docs, /metrics, /api/v1/health)
make gradio          # Gradio demo on :7860
make streamlit       # Streamlit dashboard on :8501
make audit           # pip-audit + detect-secrets + bandit
```

## Why conformal prediction?
- A normal classifier outputs **one label**. A conformal classifier outputs a **set** of labels with a mathematical guarantee that the true label is in the set with probability ≥ 1−α.
- This is **not the same** as probability calibration (ECE) — it's a stronger, distribution-free guarantee.
- Critical for medical AI: when the model is uncertain, it should say "I'm not sure" rather than guess.

## What's built
| Day | Feature |
|-----|---------|
| 0 | Scaffold, pandera schema, three-way split (60/20/20) |
| 1 | XGBoost baseline, MAPIE RAPS calibration, conformal coverage assertion |
| 2 | StandardScaler, MLflow tracking, `reports/results.json` |
| 3 | FastAPI `/predict`, `/health`, `/metrics` (Prometheus), rate limiting |
| 4 | ECE calibration, DCA, selective classification, Mondrian group fairness |
| 5 | Gradio streaming demo, HF Space deployment |
| 6 | Streamlit 4-tab dashboard: SHAP waterfall, beeswarm, coverage, group fairness |

## Documentation
- `MODEL_CARD.md` — model details, training-serving skew, fairness audit
- `MANUAL_TASKS.md` — human-only steps (HF Space create, Streamlit Cloud link)
