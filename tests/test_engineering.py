from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlforecast_realworld.data.engineering import (
    DataQualityReport,
    LogReturnTransformer,
    MarketDataEngineer,
    PercentReturnTransformer,
    PriceTransformer,
    TargetTransformerFactory,
    TargetType,
)


def test_normalize_market_frame(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
    normalized = engineer.normalize_market_frame(sample_raw_frame)
    assert "y" in normalized
    assert "sample_weight" in normalized
    assert normalized["unique_id"].nunique() == 3


def test_normalize_market_frame_log_returns(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer(target_type=TargetType.LOG_RETURN)
    normalized = engineer.normalize_market_frame(sample_raw_frame)
    assert "y" in normalized
    # Log returns should be around 0 for stable prices
    assert abs(normalized["y"].mean()) < 1.0


def test_add_calendar_features(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
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
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
    training = engineer.build_training_frame(sample_raw_frame)
    assert training["sample_weight"].min() > 0


def test_build_training_frame_has_technical_features(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
    training = engineer.build_training_frame(sample_raw_frame)
    technical_cols = {"volatility_5d", "momentum_5d", "range_pct", "volume_ma_ratio"}
    assert technical_cols.issubset(training.columns)


def test_build_static_features(sample_training_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
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
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
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
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
    report = engineer.quality_report(sample_training_frame)
    assert isinstance(report, DataQualityReport)
    assert report.rows == len(sample_training_frame)


def test_holdout_split(sample_training_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
    train, test = engineer.holdout_split(sample_training_frame, horizon=5)
    assert train["unique_id"].nunique() == test["unique_id"].nunique()
    assert len(test) == 5 * sample_training_frame["unique_id"].nunique()


def test_regularizes_missing_business_dates(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
    broken = sample_raw_frame[
        ~(
            (sample_raw_frame["unique_id"] == "AAPL.US")
            & (sample_raw_frame["ds"] == sample_raw_frame["ds"].iloc[5])
        )
    ].copy()
    training = engineer.build_training_frame(broken)
    grp = training[training["unique_id"] == "AAPL.US"].sort_values("ds")
    expected = len(pd.date_range(grp["ds"].min(), grp["ds"].max(), freq="B"))
    assert len(grp) == expected


def test_build_training_frame_invalid_row_raises(sample_raw_frame: pd.DataFrame) -> None:
    engineer = MarketDataEngineer(target_type=TargetType.PRICE)
    broken = sample_raw_frame.copy()
    broken.loc[0, "unique_id"] = " "
    with pytest.raises(ValueError):
        engineer.build_training_frame(broken)


# ============================================================================
# Target Transformer Tests
# ============================================================================

class TestPriceTransformer:
    def test_transform_returns_prices(self) -> None:
        transformer = PriceTransformer()
        prices = pd.Series([100, 105, 110])
        result = transformer.transform(prices)
        pd.testing.assert_series_equal(result, prices)

    def test_inverse_transform_returns_predictions(self) -> None:
        transformer = PriceTransformer()
        predictions = pd.Series([115, 120])
        result = transformer.inverse_transform(predictions, pd.Series([110]))
        pd.testing.assert_series_equal(result, predictions)


class TestLogReturnTransformer:
    def test_transform_calculates_log_returns(self) -> None:
        transformer = LogReturnTransformer()
        prices = pd.Series([100.0, 105.0, 110.25])
        result = transformer.transform(prices)
        expected = np.log(prices / prices.shift(1))
        pd.testing.assert_series_equal(result, expected)

    def test_log_returns_are_approximately_zero_mean(self) -> None:
        """Log returns of stable prices should be near zero."""
        transformer = LogReturnTransformer()
        # Stable prices with small random walk
        prices = pd.Series([100.0, 100.5, 100.2, 100.8, 100.3])
        result = transformer.transform(prices)
        # Mean of log returns should be small
        assert abs(result.dropna().mean()) < 0.1


class TestPercentReturnTransformer:
    def test_transform_calculates_percent_returns(self) -> None:
        transformer = PercentReturnTransformer()
        prices = pd.Series([100.0, 105.0, 110.25])
        result = transformer.transform(prices)
        expected = prices.pct_change()
        pd.testing.assert_series_equal(result, expected)


class TestTargetTransformerFactory:
    def test_create_price_transformer(self) -> None:
        transformer = TargetTransformerFactory.create(TargetType.PRICE)
        assert isinstance(transformer, PriceTransformer)

    def test_create_log_return_transformer(self) -> None:
        transformer = TargetTransformerFactory.create(TargetType.LOG_RETURN)
        assert isinstance(transformer, LogReturnTransformer)

    def test_create_percent_return_transformer(self) -> None:
        transformer = TargetTransformerFactory.create(TargetType.PERCENT_RETURN)
        assert isinstance(transformer, PercentReturnTransformer)


class TestGetLastPrices:
    def test_returns_last_price_per_series(self, sample_training_frame: pd.DataFrame) -> None:
        engineer = MarketDataEngineer(target_type=TargetType.PRICE)
        last_prices = engineer.get_last_prices(sample_training_frame)
        assert "unique_id" in last_prices.columns
        assert "last_price" in last_prices.columns
        assert len(last_prices) == sample_training_frame["unique_id"].nunique()


