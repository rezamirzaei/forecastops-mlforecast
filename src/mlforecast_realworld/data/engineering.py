from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlforecast_realworld.schemas.records import validate_market_rows

DEFAULT_SECTOR_MAP: dict[str, str] = {
    "AAPL.US": "Technology",
    "MSFT.US": "Technology",
    "GOOG.US": "Communication Services",
    "AMZN.US": "Consumer Discretionary",
    "META.US": "Communication Services",
}


@dataclass(slots=True)
class DataQualityReport:
    rows: int
    series: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_rate: float


class MarketDataEngineer:
    def __init__(
        self,
        sector_map: dict[str, str] | None = None,
        asset_class: str = "equity",
    ) -> None:
        self.sector_map = sector_map or DEFAULT_SECTOR_MAP
        self.asset_class = asset_class

    def normalize_market_frame(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        frame = raw_df.copy()
        frame["ds"] = pd.to_datetime(frame["ds"])
        frame["unique_id"] = frame["unique_id"].astype(str).str.upper()
        numeric_cols = ["open", "high", "low", "close", "volume"]
        frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["ds", "open", "high", "low", "close", "volume"])
        frame = frame.loc[(frame["close"] > 0) & (frame["volume"] > 0)].copy()
        frame = frame.drop_duplicates(subset=["unique_id", "ds"]).sort_values(["unique_id", "ds"])
        frame["y"] = frame["close"]
        frame["sector"] = frame["unique_id"].map(self.sector_map).fillna("Unknown")
        frame["asset_class"] = self.asset_class
        sector_codes = {name: idx + 1 for idx, name in enumerate(sorted(frame["sector"].unique()))}
        frame["sector_code"] = frame["sector"].map(sector_codes).astype(int)
        frame["asset_class_code"] = 1
        frame["volume"] = frame["volume"].round().astype(int)
        max_volume = frame.groupby("unique_id")["volume"].transform("max").replace(0, 1)
        frame["sample_weight"] = (frame["volume"] / max_volume).clip(lower=0.05)
        return frame.reset_index(drop=True)

    def add_calendar_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        features = frame.copy()
        ds = pd.to_datetime(features["ds"])
        features["is_weekend"] = (ds.dt.dayofweek >= 5).astype(int)
        features["is_month_start"] = ds.dt.is_month_start.astype(int)
        features["is_month_end"] = ds.dt.is_month_end.astype(int)
        features["week_of_year"] = ds.dt.isocalendar().week.astype(int)
        month = ds.dt.month
        features["month_sin"] = np.sin(2 * np.pi * month / 12)
        features["month_cos"] = np.cos(2 * np.pi * month / 12)
        return features

    def build_training_frame(
        self, raw_df: pd.DataFrame, validate_rows: bool = True
    ) -> pd.DataFrame:
        normalized = self.normalize_market_frame(raw_df)
        training_frame = self.add_calendar_features(normalized)
        if validate_rows:
            validate_market_rows(training_frame)
        return training_frame

    def build_static_features(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        static_cols = [
            "unique_id",
            "sector",
            "asset_class",
            "sector_code",
            "asset_class_code",
        ]
        return training_frame[static_cols].drop_duplicates().reset_index(drop=True)

    def build_future_exogenous(
        self,
        ids: list[str],
        last_timestamp: pd.Timestamp,
        horizon: int,
        freq: str,
    ) -> pd.DataFrame:
        future_rows: list[pd.DataFrame] = []
        for unique_id in ids:
            horizon_dates = pd.date_range(last_timestamp, periods=horizon + 1, freq=freq)[1:]
            frame = pd.DataFrame({"unique_id": unique_id, "ds": horizon_dates})
            future_rows.append(frame)
        future_df = pd.concat(future_rows, ignore_index=True)
        return self.add_calendar_features(future_df)

    def quality_report(self, frame: pd.DataFrame) -> DataQualityReport:
        return DataQualityReport(
            rows=int(len(frame)),
            series=int(frame["unique_id"].nunique()),
            start=pd.Timestamp(frame["ds"].min()),
            end=pd.Timestamp(frame["ds"].max()),
            missing_rate=float(frame.isna().mean().mean()),
        )

    def holdout_split(self, frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_parts: list[pd.DataFrame] = []
        test_parts: list[pd.DataFrame] = []
        for _, grp in frame.groupby("unique_id", sort=True):
            grp = grp.sort_values("ds")
            train_parts.append(grp.iloc[:-horizon])
            test_parts.append(grp.iloc[-horizon:])
        return (
            pd.concat(train_parts, ignore_index=True),
            pd.concat(test_parts, ignore_index=True),
        )
