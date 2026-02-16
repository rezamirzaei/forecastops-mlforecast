"""
Forecast service layer.

This module provides the ForecastService class which acts as an intermediary
between the API endpoints and the ML pipeline, handling data transformation
and business logic.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from mlforecast_realworld.api.background_tasks import (
    TaskInfo,
    TaskType,
    get_task_manager,
)
from mlforecast_realworld.config import get_settings
from mlforecast_realworld.data.sp500 import (
    SP500_TICKERS_STOOQ,
    get_all_sectors,
    get_company_info,
)
from mlforecast_realworld.ml.pipeline import ForecastPipeline
from mlforecast_realworld.schemas.records import (
    AccuracyMetric,
    ForecastRequest,
    PipelineSummary,
    forecast_records_from_frame,
)

logger = logging.getLogger(__name__)


class ForecastService:
    """Service layer for forecast operations."""

    def __init__(self, pipeline: ForecastPipeline | None = None) -> None:
        """Initialize the service with an optional pipeline instance."""
        self.pipeline = pipeline or ForecastPipeline()
        self._settings = get_settings()
        self._task_manager = get_task_manager()
        self._try_load_existing_model()

    def _try_load_existing_model(self) -> None:
        """Try to load existing model and data at startup."""
        # Load existing training data
        if self.pipeline.processed_data_path.exists():
            try:
                self.pipeline.training_frame = pd.read_parquet(
                    self.pipeline.processed_data_path
                )
                logger.info(
                    "Loaded existing training data: %d rows, %d companies",
                    len(self.pipeline.training_frame),
                    self.pipeline.training_frame["unique_id"].nunique(),
                )
            except Exception as e:
                logger.warning("Could not load training data: %s", e)
        else:
            logger.info("No existing training data found at %s", self.pipeline.processed_data_path)

        # Load existing model
        if self.pipeline.model_path.exists():
            try:
                self.pipeline.load_model()
                if self.pipeline.forecaster:
                    model_names = list(self.pipeline.forecaster.models.keys())
                else:
                    model_names = []
                logger.info("Loaded existing model: %s", ", ".join(model_names))
            except Exception as e:
                logger.warning("Could not load model: %s", e)
        else:
            logger.info("No existing model found at %s", self.pipeline.model_path)

        # Log final status
        has_data = self.pipeline.training_frame is not None
        has_model = self.pipeline.forecaster is not None
        if has_data and has_model:
            logger.info("✓ System ready for predictions")
        else:
            logger.info("System not ready: has_data=%s, has_model=%s", has_data, has_model)

    def get_status(self) -> dict[str, Any]:
        """Get current system status including task info."""
        current_task = self._task_manager.get_current_task()

        # Check if data exists (in memory or on disk)
        has_data = self.pipeline.training_frame is not None
        if not has_data and self.pipeline.processed_data_path.exists():
            # Auto-load data from disk
            try:
                self.pipeline.training_frame = pd.read_parquet(
                    self.pipeline.processed_data_path
                )
                has_data = True
                logger.info("Auto-loaded training data for status check")
            except Exception as e:
                logger.warning("Failed to load training data: %s", e)

        # Check if model exists (in memory or on disk)
        has_model = self.pipeline.forecaster is not None
        if not has_model and self.pipeline.model_path.exists():
            # Auto-load model from disk
            try:
                self.pipeline.load_model()
                has_model = True
                logger.info("Auto-loaded model for status check")
            except Exception as e:
                logger.warning("Failed to load model: %s", e)

        data_stats = {}
        if has_data and self.pipeline.training_frame is not None:
            frame = self.pipeline.training_frame
            data_stats = {
                "rows": int(len(frame)),
                "companies": int(frame["unique_id"].nunique()),
                "start_date": pd.Timestamp(frame["ds"].min()).isoformat(),
                "end_date": pd.Timestamp(frame["ds"].max()).isoformat(),
            }

        return {
            "has_data": has_data,
            "has_model": has_model,
            "is_busy": self._task_manager.is_busy(),
            "current_task": current_task.to_dict() if current_task else None,
            "data_stats": data_stats,
            "ready_for_predictions": has_data and has_model,
        }

    def start_data_update(self, tickers: list[str] | None = None) -> TaskInfo:
        """Start data update in background."""
        if self._task_manager.is_busy():
            raise ValueError("Another task is already running")

        task = self._task_manager.create_task(TaskType.DATA_UPDATE, tickers)

        def _update_data():
            self._task_manager.update_task(
                task.task_id, progress=10.0, message="Downloading data..."
            )
            self.pipeline.prepare_training_data(download=True)
            frame = self.pipeline.training_frame
            return {
                "rows": int(len(frame)) if frame is not None else 0,
                "companies": int(frame["unique_id"].nunique()) if frame is not None else 0,
            }

        self._task_manager.run_in_background(task.task_id, _update_data)
        return task

    def start_model_training(
        self, tickers: list[str] | None = None, download: bool = False
    ) -> TaskInfo:
        """Start model training in background."""
        if self._task_manager.is_busy():
            raise ValueError("Another task is already running")

        task = self._task_manager.create_task(TaskType.MODEL_TRAINING, tickers)

        def _train_model():
            self._task_manager.update_task(
                task.task_id, progress=10.0, message="Preparing training data..."
            )

            # Optionally download fresh data
            if download:
                self._task_manager.update_task(
                    task.task_id, progress=20.0, message="Downloading fresh data..."
                )
                self.pipeline.prepare_training_data(download=True)
            elif self.pipeline.training_frame is None:
                # Load existing data if available
                if self.pipeline.processed_data_path.exists():
                    self.pipeline.training_frame = pd.read_parquet(
                        self.pipeline.processed_data_path
                    )
                else:
                    raise ValueError("No training data available. Download data first.")

            frame = self.pipeline.training_frame
            if frame is None:
                raise ValueError("No training data available")

            # Filter to requested tickers if specified
            if tickers:
                frame = frame[frame["unique_id"].isin(tickers)].copy()
                if len(frame) == 0:
                    raise ValueError(f"No data found for requested tickers: {tickers}")

            self._task_manager.update_task(
                task.task_id, progress=40.0, message="Training models..."
            )
            self.pipeline.fit(frame)

            self._task_manager.update_task(
                task.task_id, progress=70.0, message="Running cross-validation..."
            )
            _, cv_summary = self.pipeline.cross_validate(frame)

            self._task_manager.update_task(
                task.task_id, progress=90.0, message="Saving model..."
            )
            self.pipeline.save_model()

            return {
                "rows": int(len(frame)),
                "companies": int(frame["unique_id"].nunique()),
                "best_model": str(cv_summary.iloc[0]["model"]) if len(cv_summary) > 0 else None,
            }

        self._task_manager.run_in_background(task.task_id, _train_model)
        return task

    def start_full_pipeline(
        self, download: bool = True, tickers: list[str] | None = None
    ) -> TaskInfo:
        """Start full pipeline (download + train) in background."""
        if self._task_manager.is_busy():
            raise ValueError("Another task is already running")

        task = self._task_manager.create_task(TaskType.FULL_PIPELINE, tickers)

        def _run_full():
            self._task_manager.update_task(
                task.task_id, progress=5.0, message="Starting pipeline..."
            )

            if download:
                self._task_manager.update_task(
                    task.task_id, progress=10.0, message="Downloading data..."
                )

            result = self.pipeline.run_full_pipeline(download=download)
            return result.get("summary", {})

        self._task_manager.run_in_background(task.task_id, _run_full)
        return task

    def get_task_status(self, task_id: str) -> TaskInfo | None:
        """Get status of a background task."""
        return self._task_manager.get_task(task_id)

    def get_all_tasks(self) -> list[TaskInfo]:
        """Get all tasks."""
        return self._task_manager.get_all_tasks()

    def get_available_series(self) -> list[str]:
        """Get list of series IDs that have training data available."""
        # First check if we have training data loaded or on disk
        frame = self.pipeline.training_frame
        if frame is None and self.pipeline.processed_data_path.exists():
            try:
                frame = pd.read_parquet(self.pipeline.processed_data_path)
            except Exception:
                frame = None

        if frame is not None:
            # Return only series that have data
            return sorted(frame["unique_id"].unique().tolist())

        # Fallback to config tickers if no data yet
        tickers = self._settings.data.tickers or list(SP500_TICKERS_STOOQ)
        unique_tickers = list(dict.fromkeys(tickers))
        return [ticker.upper() for ticker in unique_tickers]

    def get_all_companies(self) -> list[dict[str, Any]]:
        """Get S&P 500 companies that have training data available."""
        available_series = set(self.get_available_series())

        companies = []
        for ticker in SP500_TICKERS_STOOQ:
            ticker_upper = ticker.upper()
            if ticker_upper not in available_series:
                continue  # Skip companies without data

            symbol = ticker_upper.replace(".US", "")
            info = get_company_info(symbol)
            companies.append({
                "ticker": ticker_upper,
                "symbol": symbol,
                "name": info.name if info else symbol,
                "sector": info.sector if info else "Unknown",
                "has_data": True,
            })
        return companies

    def get_all_sp500_companies(self) -> list[dict[str, Any]]:
        """Get ALL S&P 500 companies (for reference, may not have data)."""
        available_series = set(self.get_available_series())

        companies = []
        for ticker in SP500_TICKERS_STOOQ:
            ticker_upper = ticker.upper()
            symbol = ticker_upper.replace(".US", "")
            info = get_company_info(symbol)
            companies.append({
                "ticker": ticker_upper,
                "symbol": symbol,
                "name": info.name if info else symbol,
                "sector": info.sector if info else "Unknown",
                "has_data": ticker_upper in available_series,
            })
        return companies

    def get_all_sectors(self) -> list[str]:
        """Get all unique sectors."""
        return get_all_sectors()

    def run_pipeline(self, download: bool = True) -> PipelineSummary:
        self.pipeline.run_full_pipeline(download=download)
        frame = self.pipeline.training_frame
        if frame is None:
            raise RuntimeError("training frame missing after pipeline run")

        trained_models = sorted(
            list(getattr(self.pipeline.forecaster, "models", {}).keys())
            if self.pipeline.forecaster is not None
            else []
        )
        return PipelineSummary(
            rows=int(len(frame)),
            unique_series=int(frame["unique_id"].nunique()),
            start=pd.Timestamp(frame["ds"].min()).to_pydatetime(),
            end=pd.Timestamp(frame["ds"].max()).to_pydatetime(),
            trained_models=trained_models,
        )

    def get_forecast(self, request: ForecastRequest) -> list[dict[str, Any]]:
        # Ensure model and data are loaded
        if self.pipeline.forecaster is None:
            if self.pipeline.model_path.exists():
                self.pipeline.load_model()
                logger.info("Loaded model for forecast request")
            else:
                raise ValueError("Model not available. Run pipeline first.")

        if self.pipeline.training_frame is None:
            if self.pipeline.processed_data_path.exists():
                self.pipeline.training_frame = pd.read_parquet(
                    self.pipeline.processed_data_path
                )
                logger.info("Loaded training data for forecast request")
            else:
                raise ValueError("Training data not available. Run pipeline first.")

        preds = self.pipeline.forecast(
            horizon=request.horizon,
            ids=request.ids,
            levels=request.levels,
        )
        model_cols = self._prediction_model_columns(preds)
        forecast_records = forecast_records_from_frame(preds, model_cols)
        return [record.model_dump() for record in forecast_records]

    def get_metrics(
        self, run_if_missing: bool = True, download: bool = False
    ) -> list[dict[str, Any]]:
        cv_summary = self.pipeline.get_cv_summary(run_if_missing=run_if_missing, download=download)
        metrics = [
            AccuracyMetric(**row).model_dump() for row in cv_summary.to_dict(orient="records")
        ]
        return metrics

    def get_history(
        self, ids: list[str] | None = None, last_n: int = 60
    ) -> list[dict[str, Any]]:
        """
        Get historical prices for visualization.

        Args:
            ids: List of series IDs (default: all available).
            last_n: Number of recent data points per series.

        Returns:
            List of historical price records.
        """
        frame = self.pipeline.training_frame

        # Auto-load data if not in memory
        if frame is None:
            if self.pipeline.processed_data_path.exists():
                frame = pd.read_parquet(self.pipeline.processed_data_path)
                self.pipeline.training_frame = frame
                logger.info("Loaded training data for history request")
            else:
                raise ValueError("No training data available. Run pipeline first.")

        # Filter by IDs if specified
        if ids:
            frame = frame[frame["unique_id"].isin(ids)]

        # Get last N records per series
        records: list[dict[str, Any]] = []
        for uid in frame["unique_id"].unique():
            series_data = frame[frame["unique_id"] == uid].sort_values("ds").tail(last_n)
            for _, row in series_data.iterrows():
                records.append({
                    "unique_id": str(row["unique_id"]),
                    "ds": pd.Timestamp(row["ds"]).isoformat(),
                    "value": float(row["close"]),
                })

        return records

    @staticmethod
    def _prediction_model_columns(preds: pd.DataFrame) -> list[str]:
        excluded = {"unique_id", "ds"}
        model_cols: list[str] = []
        for col in preds.columns:
            if col in excluded:
                continue
            if col.endswith(("-lo-80", "-hi-80", "-lo-95", "-hi-95")):
                continue
            model_cols.append(col)
        return model_cols
