from __future__ import annotations

from typing import Any

import pandas as pd

from mlforecast_realworld.ml.pipeline import ForecastPipeline
from mlforecast_realworld.schemas.records import (
    AccuracyMetric,
    ForecastRequest,
    PipelineSummary,
    forecast_records_from_frame,
)


class ForecastService:
    def __init__(self, pipeline: ForecastPipeline | None = None) -> None:
        self.pipeline = pipeline or ForecastPipeline()

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
