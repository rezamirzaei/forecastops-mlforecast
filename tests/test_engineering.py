from __future__ import annotations

import pandas as pd
import pytest

from mlforecast_realworld.data.engineering import DataQualityReport, MarketDataEngineer


def test_normalize_market_frame(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer()
    normalized = engineer.normalize_market_frame(sample_raw_frame)
    assert "y" in normalized
    assert "sample_weight" in normalized
    assert normalized["unique_id"].nunique() == 3


def test_add_calendar_features(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer()
    normalized = engineer.normalize_market_frame(sample_raw_frame)
    featured = engineer.add_calendar_features(normalized)
    expected_cols = {
        "is_weekend",
        "is_month_start",
        "is_month_end",
        "week_of_year",
        "month_sin",
        "month_cos",
    }
    assert expected_cols.issubset(featured.columns)


def test_build_training_frame_validates(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer()
    training = engineer.build_training_frame(sample_raw_frame)
    assert training["sample_weight"].min() > 0


def test_build_static_features(sample_training_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer()
    static = engineer.build_static_features(sample_training_frame)
    assert list(static.columns) == [
        "unique_id",
        "sector",
        "asset_class",
        "sector_code",
        "asset_class_code",
    ]
    assert len(static) == sample_training_frame["unique_id"].nunique()


def test_build_future_exogenous(sample_training_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer()
    ids = sorted(sample_training_frame["unique_id"].unique().tolist())
    future = engineer.build_future_exogenous(
        ids=ids,
        last_timestamp=pd.Timestamp(sample_training_frame["ds"].max()),
        horizon=5,
        freq="B",
    )
    assert len(future) == len(ids) * 5
    assert future["ds"].min() > sample_training_frame["ds"].max()


def test_quality_report(sample_training_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer()
    report = engineer.quality_report(sample_training_frame)
    assert isinstance(report, DataQualityReport)
    assert report.rows == len(sample_training_frame)


def test_holdout_split(sample_training_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer()
    train, test = engineer.holdout_split(sample_training_frame, horizon=5)
    assert train["unique_id"].nunique() == test["unique_id"].nunique()
    assert len(test) == 5 * sample_training_frame["unique_id"].nunique()


def test_build_training_frame_invalid_row_raises(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer()
    broken = sample_raw_frame.copy()
    broken.loc[0, "unique_id"] = " "
    with pytest.raises(ValueError):
        engineer.build_training_frame(broken)
