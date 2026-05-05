"""FastAPI application: conformal heart-disease prediction API (Day 5).

Routes:
  POST /api/v1/predict    -- ConformalOutput with NL summary
  GET  /api/v1/health     -- uptime, memory, n_predictions
  GET  /api/v1/model_info -- pass-through of reports/results.json
  GET  /metrics           -- Prometheus exposition (instrumentator + custom)
"""

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

import joblib
import numpy as np
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.responses import JSONResponse

from src.api.coverage_monitor import CoverageMonitor
from src.api.nl_translator import translate
from src.config import load_config
from src.data.schemas import ConformalOutput, HeartFeatures
from src.data.skew_check import check_serving_skew
from src.exceptions import PredictionError
from src.logger import get_logger
from src.models.model import ConformalXGBoost

logger = get_logger(__name__)
START_TIME = time.time()
PRED_COUNT = 0


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load model artefacts and monitoring state on startup."""
    cfg = load_config("config/config.yaml")
    app.state.cfg = cfg
    models_dir = Path(cfg["paths"]["models_dir"])
    app.state.scaler = joblib.load(models_dir / "scaler.joblib")
    app.state.model = ConformalXGBoost.load(models_dir)
    app.state.monitor = CoverageMonitor(
        window_size=cfg["monitoring"]["coverage_window_size"],
        epsilon=cfg["monitoring"]["coverage_violation_epsilon"],
    )
    yield


limiter: Limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Conformal Heart Disease AI",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])

Instrumentator().instrument(app).expose(app)


@app.exception_handler(PredictionError)
async def pred_error_handler(_: Request, exc: PredictionError) -> JSONResponse:
    """Map PredictionError to HTTP 422."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(_: Request, __: RateLimitExceeded) -> JSONResponse:
    """Map slowapi rate-limit exceeded to HTTP 429."""
    return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded."})


@app.post("/api/v1/predict", response_model=ConformalOutput)
@limiter.limit("30/minute")
async def predict(
    request: Request, body: HeartFeatures, alpha: float = 0.10
) -> ConformalOutput:
    """Run conformal prediction for a single patient feature vector.

    Returns a ConformalOutput containing the prediction set, coverage
    guarantee, mean set size, and a patient-friendly NL summary.
    """
    global PRED_COUNT
    cfg: dict[str, Any] = request.app.state.cfg
    scaler = request.app.state.scaler
    model: ConformalXGBoost = request.app.state.model
    monitor: CoverageMonitor = request.app.state.monitor

    X = scaler.transform(np.array([list(body.model_dump().values())]))
    stats_path = Path(cfg["paths"]["models_dir"]) / "training_stats.json"
    if stats_path.exists():
        skew = check_serving_skew(X, stats_path)
        if any(skew.values()):
            logger.warning("input outside training distribution: %s", skew)

    y_pred, y_ps = model.safe_predict(X, alpha=alpha)
    ps = y_ps.squeeze(-1) if y_ps.ndim == 3 else y_ps
    pred_set = [int(c) for c in np.where(ps[0])[0]]
    label = int(y_pred[0])
    mean_set_size = float(ps.sum(axis=1).mean())
    nl = translate(label=label, prediction_set=pred_set)

    PRED_COUNT += 1
    monitor.record(pred_set=pred_set, alpha=alpha)

    return ConformalOutput(
        label=label,
        prediction_set=pred_set,
        coverage_guarantee=float(1 - alpha),
        mean_set_size=mean_set_size,
        alpha_used=alpha,
        nl_summary=nl,
    )


@app.get("/api/v1/health")
async def health(request: Request) -> dict[str, Any]:
    """Return liveness and readiness metadata for monitoring dashboards."""
    return {
        "status": "ok",
        "model_loaded": request.app.state.model is not None,
        "uptime_seconds": int(time.time() - START_TIME),
        "memory_mb": int(psutil.Process().memory_info().rss / 1024 / 1024),
        "version": app.version,
        "n_predictions_served": PRED_COUNT,
    }


@app.get("/api/v1/model_info")
async def model_info(request: Request) -> dict[str, Any]:
    """Return the full evaluation results from reports/results.json."""
    p = Path(request.app.state.cfg["paths"]["results_json"])
    if not p.exists():
        raise HTTPException(status_code=404, detail="results.json not yet generated")
    return dict(json.loads(p.read_text()))
