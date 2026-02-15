"""
Forecast service layer.

This module provides the ForecastService class which acts as an intermediary
between the API endpoints and the ML pipeline, handling data transformation
and business logic.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from mlforecast_realworld.config import get_settings
from mlforecast_realworld.ml.pipeline import ForecastPipeline
from mlforecast_realworld.schemas.records import (
    AccuracyMetric,
    ForecastRequest,
    PipelineSummary,
    forecast_records_from_frame,
)


class ForecastService:
    """Service layer for forecast operations."""

    def __init__(self, pipeline: ForecastPipeline | None = None) -> None:
        """Initialize the service with an optional pipeline instance."""
        self.pipeline = pipeline or ForecastPipeline()
        self._settings = get_settings()

    def get_available_series(self) -> list[str]:
        """Get list of available series IDs for forecasting."""
        # Return configured tickers in uppercase format
        return [ticker.upper() for ticker in self._settings.data.tickers]

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
        if frame is None:
            # Try to load from disk
            if self.pipeline.processed_data_path.exists():
                frame = self.pipeline._require_training_frame()
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
