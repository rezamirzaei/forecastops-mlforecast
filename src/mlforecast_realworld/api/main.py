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
from pydantic import BaseModel, Field

from mlforecast_realworld.api.service import ForecastService
from mlforecast_realworld.config import get_settings
from mlforecast_realworld.logging_config import get_logger, setup_logging
from mlforecast_realworld.schemas.records import ForecastRequest, PipelineSummary

logger = get_logger(__name__)


class TaskStartRequest(BaseModel):
    """Request body for starting a background task."""
    download: bool = False
    tickers: list[str] | None = Field(default=None, description="Optional list of tickers to use")


class TrainingRequest(BaseModel):
    """Request body for selective model training."""
    tickers: list[str] | None = Field(
        default=None, description="Tickers to train on (all if empty)"
    )
    download: bool = Field(
        default=False, description="Whether to download fresh data first"
    )

logger = get_logger(__name__)

# API Tags for OpenAPI documentation
TAGS_METADATA = [
    {"name": "health", "description": "Health check endpoints"},
    {"name": "pipeline", "description": "ML pipeline operations (train, evaluate)"},
    {"name": "tasks", "description": "Background task management"},
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

    @app.get("/companies", tags=["forecast"])
    def available_companies() -> dict[str, Any]:
        """Get companies with training data available for forecasting."""
        companies = service.get_all_companies()
        sectors = list(set(c["sector"] for c in companies))
        return {
            "companies": companies,
            "sectors": sorted(sectors),
            "count": len(companies),
        }

    @app.get("/companies/all", tags=["forecast"])
    def all_sp500_companies() -> dict[str, Any]:
        """Get ALL S&P 500 companies (including those without data)."""
        companies = service.get_all_sp500_companies()
        sectors = service.get_all_sectors()
        available_count = sum(1 for c in companies if c["has_data"])
        return {
            "companies": companies,
            "sectors": sectors,
            "total_count": len(companies),
            "available_count": available_count,
        }

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

    @app.get("/history", tags=["forecast"])
    def history(
        ids: str | None = None,
        last_n: int = 60,
    ) -> dict[str, object]:
        """
        Get historical prices for visualization.

        Args:
            ids: Comma-separated list of series IDs (default: all).
            last_n: Number of recent data points per series (default: 60).

        Returns:
            Historical price records for charting.
        """
        try:
            id_list = [s.strip() for s in ids.split(",")] if ids else None
            records = service.get_history(ids=id_list, last_n=last_n)
            return {"records": records, "count": len(records)}
        except ValueError as exc:
            logger.warning("History error: %s", exc)
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ==================== Background Task Endpoints ====================

    @app.get("/status", tags=["health"])
    def system_status() -> dict[str, Any]:
        """Get comprehensive system status including data/model availability and task status."""
        return service.get_status()

    @app.post("/tasks/data-update", tags=["tasks"])
    def start_data_update(request: TaskStartRequest | None = None) -> dict[str, Any]:
        """
        Start background data download/update.

        This runs asynchronously and doesn't block the API.
        Use /tasks/{task_id} to check progress.
        """
        try:
            tickers = request.tickers if request else None
            task = service.start_data_update(tickers=tickers)
            logger.info("Started data update task: %s", task.task_id)
            return {"task": task.to_dict(), "message": "Data update started in background"}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/tasks/train", tags=["tasks"])
    def start_model_training(request: TrainingRequest | None = None) -> dict[str, Any]:
        """
        Start background model training.

        Optionally specify which tickers to train on.
        Training uses existing data unless download=true.
        """
        try:
            tickers = request.tickers if request else None
            download = request.download if request else False
            task = service.start_model_training(tickers=tickers, download=download)
            logger.info("Started training task: %s (tickers=%s)", task.task_id, tickers)
            return {"task": task.to_dict(), "message": "Model training started in background"}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/tasks/full-pipeline", tags=["tasks"])
    def start_full_pipeline(request: TaskStartRequest | None = None) -> dict[str, Any]:
        """
        Start full pipeline (download + train) in background.

        This is the non-blocking version of /pipeline/run.
        """
        try:
            download = request.download if request else True
            tickers = request.tickers if request else None
            task = service.start_full_pipeline(download=download, tickers=tickers)
            logger.info("Started full pipeline task: %s", task.task_id)
            return {"task": task.to_dict(), "message": "Full pipeline started in background"}
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/tasks/{task_id}", tags=["tasks"])
    def get_task_status(task_id: str) -> dict[str, Any]:
        """Get status of a background task."""
        task = service.get_task_status(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return {"task": task.to_dict()}

    @app.get("/tasks", tags=["tasks"])
    def list_tasks() -> dict[str, Any]:
        """List all background tasks (recent history)."""
        tasks = service.get_all_tasks()
        return {
            "tasks": [t.to_dict() for t in tasks],
            "count": len(tasks),
        }

    return app


app = create_app()
