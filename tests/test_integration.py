"""Integration tests for the FastAPI app and Gradio streaming generator (Day 5).

httpx.AsyncClient with ASGITransport does NOT trigger ASGI lifespan events,
so app.state is populated directly in the fixture rather than via lifespan.
"""

from typing import Any, AsyncGenerator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app
from src.api.coverage_monitor import CoverageMonitor
from src.config import load_config

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
