from __future__ import annotations

from datetime import datetime

import pandas as pd
from fastapi.testclient import TestClient

from mlforecast_realworld.api.main import create_app
from mlforecast_realworld.api.service import ForecastService
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


def test_api_endpoints(monkeypatch) -> None:
    from mlforecast_realworld.api import main as main_mod

    class DummyService:
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

    monkeypatch.setattr(main_mod, "ForecastService", lambda: DummyService())
    app = create_app()
    client = TestClient(app)

    assert client.get("/health").status_code == 200
    assert client.post("/pipeline/run", params={"download": "false"}).status_code == 200
    resp = client.post("/forecast", json={"horizon": 1, "ids": ["AAPL.US"], "levels": [80]})
    assert resp.status_code == 200
    assert resp.json()["count"] == 1
