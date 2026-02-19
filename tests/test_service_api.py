from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlforecast_realworld.api.main import create_app
from mlforecast_realworld.api.service import ForecastService
from mlforecast_realworld.config import AppSettings
from mlforecast_realworld.schemas.records import BacktestRequest, ForecastRequest


class DummyPipeline:
    def __init__(self) -> None:
        self.training_frame = pd.DataFrame(
            {
                "unique_id": ["AAPL.US", "AAPL.US"],
                "ds": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "close": [150.0, 151.0],
            }
        )
        self.forecaster = type(
            "F",
            (),
            {
                "models": {"lin_reg": object()},
                "forecast_fitted_values": lambda self: pd.DataFrame(
                    {
                        "unique_id": ["AAPL.US", "AAPL.US"],
                        "ds": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                        "lin_reg": [149.5, 150.8],
                    }
                ),
            },
        )()
        # Mock path objects that return False for exists()
        self.processed_data_path = MagicMock(spec=Path)
        self.processed_data_path.exists.return_value = False
        self.model_path = MagicMock(spec=Path)
        self.model_path.exists.return_value = False

        # data_engineer mock for backtest
        self.data_engineer = type(
            "DE", (), {"target_type": type("TT", (), {"value": "price"})()}
        )()

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


    def get_fitted_values(self, ids=None):  # noqa: ARG002
        return pd.DataFrame(
            {
                "unique_id": ["AAPL.US", "AAPL.US"],
                "ds": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                "lin_reg": [149.5, 150.8],
            }
        )
    @staticmethod
    def _add_ensemble_column(frame, model_names):
        return frame

    @staticmethod
    def _reconstruct_prices(predictions):
        return predictions


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


def test_forecast_service_get_backtest() -> None:
    """Service should return fitted values as backtest records."""
    service = ForecastService(pipeline=DummyPipeline())
    records = service.get_backtest(BacktestRequest(ids=["AAPL.US"], last_n=50))
    assert len(records) == 2  # 2 dates × 1 model
    assert records[0]["model_name"] == "lin_reg"
    assert records[0]["unique_id"] == "AAPL.US"


def test_forecast_service_get_backtest_respects_last_n() -> None:
    """Backtest should respect last_n parameter."""
    service = ForecastService(pipeline=DummyPipeline())
    records = service.get_backtest(BacktestRequest(ids=["AAPL.US"], last_n=1))
    assert len(records) == 1  # Only last date
    assert records[0]["value"] == 150.8  # Last fitted value


def test_forecast_service_get_backtest_does_not_require_model_load() -> None:
    pipeline = DummyPipeline()
    pipeline.forecaster = None
    pipeline.model_path.exists.return_value = True

    def failing_load_model():
        raise RuntimeError("model load should not be called for backtest")

    pipeline.load_model = failing_load_model
    service = ForecastService(pipeline=pipeline)

    records = service.get_backtest(BacktestRequest(ids=["AAPL.US"], last_n=50))
    assert len(records) == 2


def test_forecast_service_backtest_reconstructs_return_target_prices() -> None:
    pipeline = DummyPipeline()
    pipeline.data_engineer = type(
        "DE", (), {"target_type": type("TT", (), {"value": "log_return"})()}
    )()

    def returns_fitted_values(ids=None):  # noqa: ARG001
        frame = pd.DataFrame(
            {
                "unique_id": ["AAPL.US", "AAPL.US"],
                "ds": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")],
                # Day 1: 0% change, day 2: +10% change
                "lin_reg": [0.0, 0.0953101798],
            }
        )
        if ids:
            frame = frame[frame["unique_id"].isin(ids)]
        return frame

    pipeline.get_fitted_values = returns_fitted_values
    service = ForecastService(pipeline=pipeline)

    records = service.get_backtest(BacktestRequest(ids=["AAPL.US"], last_n=2))
    by_date = {row["ds"]: row["value"] for row in records}
    assert by_date[pd.Timestamp("2024-01-01").to_pydatetime()] == 150.0
    assert by_date[pd.Timestamp("2024-01-02").to_pydatetime()] == pytest.approx(165.0)


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

        def get_history(self, ids=None, last_n: int = 60):  # noqa: ARG002
            return [
                {
                    "unique_id": "AAPL.US",
                    "ds": datetime(2024, 1, 31).isoformat(),
                    "value": 150.0,
                }
            ]

        def get_backtest(self, request):  # noqa: ARG002
            return [
                {
                    "unique_id": "AAPL.US",
                    "ds": datetime(2024, 1, 31).isoformat(),
                    "model_name": "lin_reg",
                    "value": 149.5,
                }
            ]

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

    # Test backtest endpoint
    backtest_resp = client.get("/backtest", params={"ids": "AAPL.US", "last_n": "50"})
    assert backtest_resp.status_code == 200
    assert backtest_resp.json()["count"] == 1
    assert backtest_resp.json()["records"][0]["model_name"] == "lin_reg"
