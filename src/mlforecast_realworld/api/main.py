"""
FastAPI application for MLForecast Real-World API.

This module provides REST endpoints for running ML pipelines,
generating forecasts, and retrieving model metrics.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mlforecast_realworld.api.service import ForecastService
from mlforecast_realworld.config import get_settings
from mlforecast_realworld.logging_config import get_logger, setup_logging
from mlforecast_realworld.schemas.records import ForecastRequest, PipelineSummary

logger = get_logger(__name__)

# API Tags for OpenAPI documentation
TAGS_METADATA = [
    {"name": "health", "description": "Health check endpoints"},
    {"name": "pipeline", "description": "ML pipeline operations (train, evaluate)"},
    {"name": "forecast", "description": "Generate predictions"},
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    settings = get_settings()
    setup_logging(environment=settings.environment)
    logger.info("MLForecast API starting up in %s mode", settings.environment)
    yield
    logger.info("MLForecast API shutting down...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="MLForecast Real-World API",
        version="0.1.0",
        description=(
            "Production forecasting API powered by mlforecast and real market data. "
            "Supports multi-model ensemble forecasting with prediction intervals."
        ),
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
    )

    # CORS middleware for frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:4200", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = ForecastService()

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Global exception handler for unhandled errors."""
        logger.exception("Unhandled error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    @app.get("/health", tags=["health"])
    def health() -> dict[str, Any]:
        """Health check endpoint with service status."""
        return {
            "status": "ok",
            "service": "mlforecast-api",
            "version": "0.1.0",
            "model_loaded": service.pipeline.forecaster is not None,
        }

    @app.get("/series", tags=["forecast"])
    def available_series() -> dict[str, Any]:
        """Get list of available series for forecasting."""
        series = service.get_available_series()
        return {"series": series, "count": len(series)}

    @app.post("/pipeline/run", response_model=PipelineSummary, tags=["pipeline"])
    def run_pipeline(download: bool = True) -> PipelineSummary:
        """Run the full ML pipeline: download data, engineer features, train models."""
        logger.info("Running pipeline with download=%s", download)
        result = service.run_pipeline(download=download)
        logger.info("Pipeline completed: %d rows, %d series", result.rows, result.unique_series)
        return result

    @app.get("/pipeline/metrics", tags=["pipeline"])
    def pipeline_metrics(
        run_if_missing: bool = True, download: bool = False
    ) -> dict[str, object]:
        """Get cross-validation metrics for all trained models."""
        metrics = service.get_metrics(run_if_missing=run_if_missing, download=download)
        best = min(metrics, key=lambda m: m["smape"])["model"] if metrics else None
        return {"metrics": metrics, "best_model": best, "count": len(metrics)}

    @app.post("/forecast", tags=["forecast"])
    def forecast(request: ForecastRequest) -> dict[str, object]:
        """Generate forecasts for specified series and horizon."""
        try:
            logger.info("Generating forecast: horizon=%d, ids=%s", request.horizon, request.ids)
            records = service.get_forecast(request)
            return {"records": records, "count": len(records)}
        except ValueError as exc:
            logger.warning("Forecast validation error: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.error("Runtime error during forecast: %s", exc)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
