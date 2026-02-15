from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
from fastapi.testclient import TestClient

from mlforecast_realworld.api.main import create_app
from mlforecast_realworld.api.service import ForecastService
from mlforecast_realworld.config import AppSettings
from mlforecast_realworld.schemas.records import ForecastRequest


class DummyPipeline:
    def __init__(self) -> None:
        self.training_frame = pd.DataFrame(
            {
                "unique_id": ["AAPL.US", "AAPL.US"],
                "ds": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
            }
        )
        self.forecaster = type("F", (), {"models": {"lin_reg": object()}})()

    def run_full_pipeline(self, download: bool = True):  # noqa: ARG002
        return {"summary": {"rows": 2}}

    def forecast(self, horizon: int, ids=None, levels=None):  # noqa: ARG002
        return pd.DataFrame(
            {
                "unique_id": ["AAPL.US"],
                "ds": [pd.Timestamp("2024-01-03")],
                "lin_reg": [123.4],
                "lin_reg-lo-80": [120.0],
                "lin_reg-hi-80": [126.0],
            }
        )

    def get_cv_summary(self, run_if_missing: bool = False, download: bool = False):  # noqa: ARG002
        return pd.DataFrame([{"model": "lin_reg", "smape": 1.23, "wape": 1.11}])


def test_forecast_service_run_pipeline() -> None:
    service = ForecastService(pipeline=DummyPipeline())
    summary = service.run_pipeline(download=False)
    assert summary.rows == 2
    assert summary.trained_models == ["lin_reg"]


def test_forecast_service_get_forecast() -> None:
    service = ForecastService(pipeline=DummyPipeline())
    records = service.get_forecast(ForecastRequest(horizon=1))
    assert len(records) == 1
    assert records[0]["model_name"] == "lin_reg"


def test_forecast_service_get_metrics() -> None:
    service = ForecastService(pipeline=DummyPipeline())
    metrics = service.get_metrics(run_if_missing=True)
    assert len(metrics) == 1
    assert metrics[0]["model"] == "lin_reg"


def test_forecast_service_series_uses_training_data_when_available() -> None:
    """Service should return only series that have training data."""
    service = ForecastService(pipeline=DummyPipeline())
    series = service.get_available_series()
    # DummyPipeline has training_frame with AAPL.US
    assert "AAPL.US" in series
    # Should only return series from training data, not all S&P 500
    assert len(series) <= 10  # Training data has limited series


def test_forecast_service_series_falls_back_to_config_when_no_data() -> None:
    """Service should fallback to config tickers when no training data."""
    pipeline = DummyPipeline()
    pipeline.training_frame = None  # No training data
    pipeline.processed_data_path = MagicMock()
    pipeline.processed_data_path.exists.return_value = False

    service = ForecastService(pipeline=pipeline)
    service._settings = AppSettings(data={"tickers": ["aapl.us", "msft.us"]})
    series = service.get_available_series()
    # Falls back to config tickers
    assert "AAPL.US" in series
    assert "MSFT.US" in series


def test_api_endpoints(monkeypatch) -> None:
    from mlforecast_realworld.api import main as main_mod

    class DummyPipelineForService:
        forecaster = None

    class DummyService:
        def __init__(self):
            self.pipeline = DummyPipelineForService()

        def get_available_series(self):
            return ["AAPL.US", "MSFT.US"]

        def run_pipeline(self, download: bool = True):  # noqa: ARG002
            from mlforecast_realworld.schemas.records import PipelineSummary

            return PipelineSummary(
                rows=10,
                unique_series=2,
                start=datetime(2024, 1, 1),
                end=datetime(2024, 1, 31),
                trained_models=["lin_reg"],
            )

        def get_forecast(self, request: ForecastRequest):  # noqa: ARG002
            return [
                {
                    "unique_id": "AAPL.US",
                    "ds": datetime(2024, 2, 1),
                    "model_name": "lin_reg",
                    "value": 1.0,
                }
            ]

        def get_metrics(self, run_if_missing: bool = True, download: bool = False):  # noqa: ARG002
            return [{"model": "lin_reg", "smape": 1.2, "wape": 1.1}]

    monkeypatch.setattr(main_mod, "ForecastService", lambda: DummyService())
    app = create_app()
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.post("/pipeline/run", params={"download": "false"}).status_code == 200
    metrics = client.get("/pipeline/metrics", params={"run_if_missing": "true"})
    assert metrics.status_code == 200
    assert metrics.json()["best_model"] == "lin_reg"
    resp = client.post("/forecast", json={"horizon": 1, "ids": ["AAPL.US"], "levels": [80]})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
