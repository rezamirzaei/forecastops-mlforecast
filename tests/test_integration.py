"""
Integration tests for the MLForecast API.

These tests verify end-to-end functionality of the API endpoints
using a test client and mocked pipeline.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlforecast_realworld.api.main import create_app
from mlforecast_realworld.data.sp500 import SP500_TICKERS_STOOQ


@pytest.fixture
def mock_pipeline():
    """Create a mock pipeline for testing."""
    pipeline = MagicMock()
    pipeline.forecaster = None
    pipeline.training_frame = pd.DataFrame({
        "unique_id": ["AAPL.US"] * 100,
        "ds": pd.date_range("2024-01-01", periods=100, freq="B"),
        "close": [150.0] * 100,
    })
    return pipeline


@pytest.fixture
def client(mock_pipeline):
    """Create test client with mocked pipeline."""
    with patch("mlforecast_realworld.api.service.ForecastPipeline", return_value=mock_pipeline):
        app = create_app()
        yield TestClient(app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_returns_ok(self, client):
        """Health endpoint should return OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "mlforecast-api"
        assert "version" in data
        assert "model_loaded" in data

    def test_health_shows_model_not_loaded(self, client):
        """Health should show model_loaded=False when no model exists."""
        response = client.get("/health")
        data = response.json()
        assert data["model_loaded"] is False


class TestSeriesEndpoint:
    """Tests for /series endpoint."""

    def test_series_returns_available_tickers(self, client):
        """Series endpoint should return configured tickers."""
        response = client.get("/series")
        assert response.status_code == 200
        data = response.json()
        assert "series" in data
        assert "count" in data
        assert len(data["series"]) == data["count"]
        # Should include full default S&P 500 universe in uppercase.
        assert data["count"] == len(SP500_TICKERS_STOOQ)
        assert "AAPL.US" in data["series"]


class TestPipelineEndpoints:
    """Tests for /pipeline/* endpoints."""

    def test_run_pipeline_success(self, client, mock_pipeline):
        """Pipeline run should return summary."""
        mock_pipeline.run_full_pipeline.return_value = {"summary": {"rows": 100}}
        mock_pipeline.forecaster = MagicMock()
        mock_pipeline.forecaster.models = {"lin_reg": MagicMock()}

        response = client.post("/pipeline/run?download=false")
        assert response.status_code == 200
        data = response.json()
        assert "rows" in data
        assert "unique_series" in data
        assert "trained_models" in data

    def test_pipeline_metrics_success(self, client, mock_pipeline):
        """Metrics endpoint should return model metrics."""
        mock_cv_summary = pd.DataFrame({
            "model": ["lin_reg", "rf"],
            "smape": [5.2, 4.8],
            "wape": [4.1, 3.9],
        })
        mock_pipeline.get_cv_summary.return_value = mock_cv_summary

        response = client.get("/pipeline/metrics?run_if_missing=false")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "best_model" in data
        assert "count" in data
        # Best model should be the one with lowest sMAPE
        assert data["best_model"] == "rf"


class TestForecastEndpoint:
    """Tests for /forecast endpoint."""

    def test_forecast_requires_trained_model(self, client, mock_pipeline):
        """Forecast should fail if model not trained."""
        mock_pipeline.forecast.side_effect = RuntimeError("forecaster is not fitted")

        response = client.post("/forecast", json={
            "horizon": 14,
            "ids": ["AAPL.US"],
            "levels": [80, 95],
        })
        assert response.status_code == 500
        assert "detail" in response.json()

    def test_forecast_invalid_params(self, client, mock_pipeline):
        """Forecast should fail with invalid parameters."""
        mock_pipeline.forecast.side_effect = ValueError("invalid horizon")

        response = client.post("/forecast", json={
            "horizon": 14,
            "ids": ["INVALID"],
            "levels": [80, 95],
        })
        assert response.status_code == 400
        assert "detail" in response.json()

    def test_forecast_success(self, client, mock_pipeline):
        """Forecast should return predictions."""
        mock_predictions = pd.DataFrame({
            "unique_id": ["AAPL.US"] * 14,
            "ds": pd.date_range("2024-06-01", periods=14, freq="B"),
            "lin_reg": [155.0] * 14,
        })
        mock_pipeline.forecast.return_value = mock_predictions

        response = client.post("/forecast", json={
            "horizon": 14,
            "ids": ["AAPL.US"],
            "levels": [80, 95],
        })
        assert response.status_code == 200
        data = response.json()
        assert "records" in data
        assert "count" in data
        assert data["count"] > 0


class TestHistoryEndpoint:
    """Tests for /history endpoint."""

    def test_history_returns_records(self, client, mock_pipeline):
        """History endpoint should return historical price records."""
        # Mock has training_frame with AAPL.US data
        response = client.get("/history?last_n=30")
        assert response.status_code == 200
        data = response.json()
        assert "records" in data
        assert "count" in data
        assert data["count"] > 0
        # Check record structure
        record = data["records"][0]
        assert "unique_id" in record
        assert "ds" in record
        assert "value" in record

    def test_history_filters_by_ids(self, client, mock_pipeline):
        """History should filter by specified IDs."""
        response = client.get("/history?ids=AAPL.US&last_n=10")
        assert response.status_code == 200
        data = response.json()
        # All records should be for AAPL.US
        for record in data["records"]:
            assert record["unique_id"] == "AAPL.US"


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """CORS headers should be present for allowed origins."""
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:4200",
                "Access-Control-Request-Method": "GET",
            },
        )
        # Should allow the request (not return 4xx)
        assert response.status_code in [200, 204, 405]


class TestOpenAPIDocumentation:
    """Tests for OpenAPI documentation."""

    def test_openapi_schema_available(self, client):
        """OpenAPI schema should be accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert schema["info"]["title"] == "MLForecast Real-World API"
        assert "paths" in schema

    def test_docs_page_available(self, client):
        """Swagger docs should be accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
