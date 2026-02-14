from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

import pandas as pd
from pydantic import BaseModel, Field, PositiveFloat, PositiveInt, ValidationError, field_validator


class MarketRecord(BaseModel):
    unique_id: str
    ds: datetime
    open: PositiveFloat
    high: PositiveFloat
    low: PositiveFloat
    close: PositiveFloat
    volume: PositiveInt
    y: PositiveFloat
    sample_weight: PositiveFloat
    sector: str
    asset_class: str
    sector_code: PositiveInt
    asset_class_code: PositiveInt
    is_weekend: int = Field(ge=0, le=1)
    is_month_start: int = Field(ge=0, le=1)
    is_month_end: int = Field(ge=0, le=1)
    week_of_year: int = Field(ge=1, le=53)

    @field_validator("unique_id", "sector", "asset_class")
    @classmethod
    def non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be empty")
        return stripped


class ForecastRequest(BaseModel):
    horizon: PositiveInt = 14
    ids: list[str] | None = None
    levels: list[int] = Field(default_factory=lambda: [80, 95])


class PipelineSummary(BaseModel):
    rows: int
    unique_series: int
    start: datetime
    end: datetime
    trained_models: list[str]


class AccuracyMetric(BaseModel):
    model: str
    smape: float
    wape: float


class ForecastRecord(BaseModel):
    unique_id: str
    ds: datetime
    model_name: str
    value: float


def validate_market_rows(df: pd.DataFrame, sample_size: int = 500) -> None:
    if df.empty:
        raise ValueError("training frame is empty")
    sample = df.head(sample_size)
    errors: list[ValidationError] = []
    for row in sample.to_dict(orient="records"):
        try:
            MarketRecord.model_validate(row)
        except ValidationError as exc:  # pragma: no cover - exercised via tests with invalid rows
            errors.append(exc)
    if errors:
        raise ValueError(f"market row validation failed for {len(errors)} rows")


def forecast_records_from_frame(
    df: pd.DataFrame,
    model_columns: Iterable[str],
) -> list[ForecastRecord]:
    records: list[ForecastRecord] = []
    for _, row in df.iterrows():
        for model_name in model_columns:
            records.append(
                ForecastRecord(
                    unique_id=str(row["unique_id"]),
                    ds=pd.Timestamp(row["ds"]).to_pydatetime(),
                    model_name=model_name,
                    value=float(row[model_name]),
                )
            )
    return records
