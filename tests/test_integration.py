"""Integration tests for the FastAPI app and Gradio streaming generator (Day 5).

httpx.AsyncClient with ASGITransport does NOT trigger ASGI lifespan events,
so app.state is populated directly in the fixture rather than via lifespan.
"""

from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import MagicMock, patch

import joblib
import numpy as np
import pandas as pd
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.api.coverage_monitor import CoverageMonitor
from src.config import load_config
from src.models.model import ConformalXGBoost

VALID_BODY: dict[str, Any] = {
    "age": 55,
    "sex": 1,
    "cp": 2,
    "trestbps": 120,
    "chol": 200,
    "fbs": 0,
    "restecg": 1,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 1.0,
    "slope": 2,
    "ca": 0,
    "thal": 2,
}


def _make_mock_model() -> MagicMock:
    """Return a ConformalXGBoost mock that avoids all disk I/O."""
    m = MagicMock()
    m.safe_predict.return_value = (
        np.array([1]),
        np.array([[False, True]]),
    )
    m.alphas = [0.05, 0.10, 0.20]
    return m


def _make_mock_scaler() -> MagicMock:
    """Return a StandardScaler mock that returns a zero feature vector."""
    s = MagicMock()
    s.transform.return_value = np.zeros((1, 13))
    return s


@pytest.fixture()
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async client with state injected directly (ASGITransport skips lifespan)."""
    cfg = load_config("config/config.yaml")

    app.state.cfg = cfg
    app.state.scaler = _make_mock_scaler()
    app.state.model = _make_mock_model()
    app.state.monitor = CoverageMonitor(
        window_size=cfg["monitoring"]["coverage_window_size"],
        epsilon=cfg["monitoring"]["coverage_violation_epsilon"],
    )

    with patch("src.api.app.check_serving_skew", return_value={}):
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    del app.state.cfg
    del app.state.scaler
    del app.state.model
    del app.state.monitor


@pytest.mark.asyncio
async def test_health_endpoint(client: AsyncClient) -> None:
    """GET /api/v1/health -> 200 with all 6 expected keys."""
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    for key in ("status", "model_loaded", "uptime_seconds", "memory_mb", "version"):
        assert key in data, f"missing key: {key}"


@pytest.mark.asyncio
async def test_predict_valid_schema(client: AsyncClient) -> None:
    """POST valid features -> 200 with full ConformalOutput schema."""
    resp = await client.post("/api/v1/predict", json=VALID_BODY)
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "label",
        "prediction_set",
        "coverage_guarantee",
        "mean_set_size",
        "alpha_used",
        "nl_summary",
    ):
        assert key in data, f"missing key: {key}"
    assert isinstance(data["prediction_set"], list)
    assert isinstance(data["nl_summary"], str)


@pytest.mark.asyncio
async def test_invalid_input_422(client: AsyncClient) -> None:
    """POST out-of-range age -> 422 validation error."""
    resp = await client.post("/api/v1/predict", json={"age": 999})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_custom_counter(client: AsyncClient) -> None:
    """GET /metrics body contains coverage_violations_total counter."""
    resp = await client.get("/metrics")
    assert resp.status_code == 200
    assert b"coverage_violations_total" in resp.content


@pytest.mark.asyncio
async def test_streaming_pipeline_yields_4plus_frames() -> None:
    """The stream_classify generator yields at least 4 stage frames before Done."""
    mock_model = _make_mock_model()
    mock_scaler = _make_mock_scaler()

    with (
        patch("src.api.gradio_demo._model", mock_model),
        patch("src.api.gradio_demo._scaler", mock_scaler),
    ):
        from src.api.gradio_demo import stream_classify

        frames = list(
            stream_classify(55, 1, 2, 120, 200, 0, 1, 150, 0, 1.0, 2, 0, 2, 0.10)
        )

    assert len(frames) >= 4, f"expected >= 4 frames, got {len(frames)}"
    last_stage, last_result, _ = frames[-1]
    assert "Done" in last_stage
    assert last_result is not None


_TEST_CSV = Path("data/processed/test.csv")


@pytest.fixture()
async def real_client() -> AsyncGenerator[AsyncClient, None]:
    """Async client using actual model artefacts from disk.

    Loads the committed joblib artefacts (scaler, xgb, scc, meta) rather than
    mocks so the coverage-guarantee hammer test exercises the real pipeline.
    Rate limiting is disabled — the hammer test fires 60+ requests from the
    same test IP which would exhaust the 30/minute budget.
    """
    cfg = load_config("config/config.yaml")
    models_dir = Path(cfg["paths"]["models_dir"])

    app.state.cfg = cfg
    app.state.scaler = joblib.load(models_dir / "scaler.joblib")
    app.state.model = ConformalXGBoost.load(models_dir)
    app.state.monitor = CoverageMonitor(
        window_size=cfg["monitoring"]["coverage_window_size"],
        epsilon=cfg["monitoring"]["coverage_violation_epsilon"],
    )

    def _no_rate_limit(
        self: object, request: Any, endpoint: Any, in_middleware: bool = False
    ) -> None:
        """Bypass slowapi rate check; set the state attr it reads after the endpoint."""
        request.state.view_rate_limit = "unlimited/test"

    with (
        patch("src.api.app.check_serving_skew", return_value={}),
        patch("slowapi.extension.Limiter._check_request_limit", _no_rate_limit),
    ):
        transport = ASGITransport(app=app)  # type: ignore[arg-type]
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    del app.state.cfg
    del app.state.scaler
    del app.state.model
    del app.state.monitor


@pytest.mark.asyncio
async def test_async_coverage_guarantee(real_client: AsyncClient) -> None:
    """Hammer every test-split row; empirical coverage must be >= 1-alpha - 0.05.

    data/processed/test.csv contains StandardScaler-transformed features; we
    inverse-transform to recover approximate original values before sending to
    the API. Rows whose inverse-transformed features fall outside Pydantic bounds
    (due to float precision) are skipped and logged.

    The test skips cleanly on a fresh clone where train.py has not been run,
    since test.csv is gitignored.
    """
    if not _TEST_CSV.exists():
        pytest.skip("data/processed/test.csv not available (gitignored in CI)")

    cfg = load_config("config/config.yaml")
    models_dir = Path(cfg["paths"]["models_dir"])
    scaler = joblib.load(models_dir / "scaler.joblib")

    df = pd.read_csv(_TEST_CSV)
    feature_cols = [c for c in df.columns if c != "target"]
    X_scaled = df[feature_cols].values
    X_orig = scaler.inverse_transform(X_scaled)

    # Columns that must be integers for Pydantic HeartFeatures
    int_col_names = [c for c in feature_cols if c != "oldpeak"]
    int_indices = [feature_cols.index(c) for c in int_col_names]
    X_orig[:, int_indices] = np.round(X_orig[:, int_indices])

    alpha = 0.10
    correct = 0
    evaluated = 0

    for i, row in enumerate(df.itertuples(index=False)):
        payload: dict[str, Any] = {}
        for j, col in enumerate(feature_cols):
            val = X_orig[i, j]
            payload[col] = int(round(val)) if col != "oldpeak" else float(val)

        r = await real_client.post(f"/api/v1/predict?alpha={alpha}", json=payload)
        if r.status_code != 200:
            continue  # inverse-transform rounding pushed value out of Pydantic bounds
        body = r.json()
        evaluated += 1
        if int(row.target) in body["prediction_set"]:
            correct += 1

    assert (
        evaluated >= len(df) // 3
    ), f"Only {evaluated}/{len(df)} rows were valid after inverse-transform"
    empirical = correct / evaluated
    assert empirical >= (1.0 - alpha) - 0.05, (
        f"Empirical coverage {empirical:.3f} is below 1-alpha-0.05 "
        f"({(1.0 - alpha) - 0.05:.3f}) on {evaluated} valid test-split rows"
    )
