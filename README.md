# Conformal Prediction · Uncertainty-Aware Medical AI

> Production-grade medical AI with mathematical coverage guarantees.
> UCI Heart Disease · MAPIE (RAPS, cv='prefit') · XGBoost · FastAPI · Gradio · Streamlit · MLflow.

**Status:** Day 0 scaffold. Live URLs added on Day 7.

## Live demos
- 🔗 Gradio (HuggingFace Spaces) — *added on Day 5*
- 🔗 Streamlit Cloud dashboard — *added on Day 6*
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
*Pedagogy section finalised on Day 7 (rule F).* Three-line preview:
- A normal classifier outputs **one label**. A conformal classifier outputs a **set** of labels with a mathematical guarantee that the true label is in the set with probability ≥ 1−α.
- This is **not the same** as probability calibration (ECE) — it's a stronger, distribution-free guarantee.
- Critical for medical AI: when the model is uncertain, it should say "I'm not sure" rather than guess.

## Documentation
- `MODEL_CARD.md` — model details, training-serving skew, fairness audit
- `MANUAL_TASKS.md` — human-only steps (HF Space create, Streamlit Cloud link)
- `research-notes/` — reading log + design rationale
- `docs/blog/conformal-prediction-in-medical-ai.md` — long-form blog post
