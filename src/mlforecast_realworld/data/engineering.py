"""
Market data engineering module.

This module provides the MarketDataEngineer class for transforming raw market
data into a format suitable for MLForecast training. It handles:
- Data normalization and validation
- Frequency regularization (filling gaps in market calendars)
- Returns calculation (log returns for better stationarity)
- Calendar feature generation
- Train/test splitting for time-series
- Price reconstruction from predicted returns

Example:
    >>> engineer = MarketDataEngineer()
    >>> training_df = engineer.build_training_frame(raw_df)
    >>> train, test = engineer.holdout_split(training_df, horizon=14)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
import pandas as pd

from mlforecast_realworld.schemas.records import validate_market_rows

# ============================================================================
# Type Definitions
# ============================================================================

class TargetType(Enum):
    """Target variable type for prediction."""
    PRICE = "price"
    LOG_RETURN = "log_return"
    PERCENT_RETURN = "percent_return"


class TargetTransformer(Protocol):
    """Protocol for target variable transformers."""

    def transform(self, prices: pd.Series) -> pd.Series:
        """Transform prices to target variable."""
        ...

    def inverse_transform(
        self, predictions: pd.Series, last_prices: pd.Series
    ) -> pd.Series:
        """Reconstruct prices from predicted values."""
        ...


# ============================================================================
# Target Transformers (Strategy Pattern)
# ============================================================================

class PriceTransformer:
    """Direct price prediction (no transformation)."""

    def transform(self, prices: pd.Series) -> pd.Series:
        """Return prices as-is."""
        return prices.copy()

    def inverse_transform(
        self, predictions: pd.Series, last_prices: pd.Series  # noqa: ARG002
    ) -> pd.Series:
        """Return predictions as-is (already prices)."""
        return predictions.copy()


class LogReturnTransformer:
    """
    Log return transformation for better stationarity.

    Log returns: r_t = ln(P_t / P_{t-1})
    Reconstruction: P_t = P_{t-1} * exp(r_t)

    Benefits:
    - More stationary than raw prices
    - Symmetric for gains/losses
    - Additive over time periods
    - Better for percentage-based metrics
    """

    def transform(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns from prices."""
        return np.log(prices / prices.shift(1))

    def inverse_transform(
        self, predictions: pd.Series, last_prices: pd.Series
    ) -> pd.Series:
        """
        Reconstruct prices from log return predictions.

        Args:
            predictions: Predicted log returns for each step.
            last_prices: Last known price for each series.

        Returns:
            Reconstructed price series.
        """
        # Cumulative sum of log returns, then exponentiate
        cumulative_log_returns = predictions.groupby(level=0).cumsum()
        return last_prices * np.exp(cumulative_log_returns)


class PercentReturnTransformer:
    """
    Percent return transformation.

    Percent returns: r_t = (P_t - P_{t-1}) / P_{t-1}
    Reconstruction: P_t = P_{t-1} * (1 + r_t)
    """

    def transform(self, prices: pd.Series) -> pd.Series:
        """Calculate percent returns from prices."""
        return prices.pct_change()

    def inverse_transform(
        self, predictions: pd.Series, last_prices: pd.Series
    ) -> pd.Series:
        """Reconstruct prices from percent return predictions."""
        cumulative_returns = (1 + predictions).groupby(level=0).cumprod()
        return last_prices * cumulative_returns


# ============================================================================
# Factory for Target Transformers
# ============================================================================

class TargetTransformerFactory:
    """Factory for creating target transformers."""

    _transformers: dict[TargetType, type] = {
        TargetType.PRICE: PriceTransformer,
        TargetType.LOG_RETURN: LogReturnTransformer,
        TargetType.PERCENT_RETURN: PercentReturnTransformer,
    }

    @classmethod
    def create(cls, target_type: TargetType) -> TargetTransformer:
        """Create a target transformer instance."""
        transformer_class = cls._transformers.get(target_type)
        if transformer_class is None:
            raise ValueError(f"Unknown target type: {target_type}")
        return transformer_class()

    @classmethod
    def register(cls, target_type: TargetType, transformer_class: type) -> None:
        """Register a new transformer type."""
        cls._transformers[target_type] = transformer_class


# ============================================================================
# Constants
# ============================================================================

DEFAULT_SECTOR_MAP: dict[str, str] = {
    "AAPL.US": "Technology",
    "MSFT.US": "Technology",
    "GOOG.US": "Communication Services",
    "AMZN.US": "Consumer Discretionary",
    "META.US": "Communication Services",
}


# ============================================================================
# Data Classes
# ============================================================================

@dataclass(slots=True)
class DataQualityReport:
    """Data quality metrics for a training frame."""

    rows: int
    series: int
    start: pd.Timestamp
    end: pd.Timestamp
    missing_rate: float
    target_type: str = "price"


# ============================================================================
# Main Engineer Class
# ============================================================================

class MarketDataEngineer:
    """
    Engineer raw market data into MLForecast-ready training frames.

    This class handles the complete data preparation workflow:
    - Normalization: type conversion, deduplication, validation
    - Frequency regularization: fill gaps in market calendars
    - Target transformation: price, log returns, or percent returns
    - Feature engineering: calendar features, sector codes
    - Quality reporting: missing rates, date ranges

    Attributes:
        sector_map: Mapping of ticker symbols to sector names.
        asset_class: Asset class label (default: "equity").
        freq: Pandas frequency string (default: "B" for business days).
        target_type: Type of target variable (price, log_return, percent_return).
        target_transformer: Transformer for target variable.
    """

    def __init__(
        self,
        sector_map: dict[str, str] | None = None,
        asset_class: str = "equity",
        freq: str = "B",
        target_type: TargetType = TargetType.LOG_RETURN,
    ) -> None:
        self.sector_map = sector_map or DEFAULT_SECTOR_MAP
        self.asset_class = asset_class
        self.freq = freq
        self.target_type = target_type
        self.target_transformer = TargetTransformerFactory.create(target_type)

    def _compute_target(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Compute target variable based on target_type."""
        result = frame.copy()

        if self.target_type == TargetType.PRICE:
            result["y"] = result["close"]
        else:
            # Calculate returns per series
            result["y"] = result.groupby("unique_id")["close"].transform(
                self.target_transformer.transform
            )
            # Fill first NaN with 0 (no return on first observation)
            result["y"] = result["y"].fillna(0)

        return result

    def reconstruct_prices(
        self,
        predictions: pd.DataFrame,
        last_known_prices: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Reconstruct prices from predicted returns.

        Args:
            predictions: DataFrame with columns [unique_id, ds, model_predictions...]
            last_known_prices: DataFrame with [unique_id, last_price]

        Returns:
            DataFrame with reconstructed prices.
        """
        if self.target_type == TargetType.PRICE:
            return predictions.copy()

        result = predictions.copy()
        price_map = last_known_prices.set_index("unique_id")["last_price"]

        # Get model columns (excluding unique_id, ds, and interval columns)
        model_cols = [
            col for col in result.columns
            if col not in ("unique_id", "ds")
            and not col.endswith(("-lo-80", "-hi-80", "-lo-95", "-hi-95"))
        ]

        for model_col in model_cols:
            for uid in result["unique_id"].unique():
                mask = result["unique_id"] == uid
                last_price = price_map.get(uid, 1.0)
                pred_returns = result.loc[mask, model_col]

                if self.target_type == TargetType.LOG_RETURN:
                    # Cumulative log return -> price
                    cum_log_return = pred_returns.cumsum()
                    result.loc[mask, model_col] = last_price * np.exp(cum_log_return)
                else:  # PERCENT_RETURN
                    # Cumulative percent return -> price
                    cum_return = (1 + pred_returns).cumprod()
                    result.loc[mask, model_col] = last_price * cum_return

        return result

    def normalize_market_frame(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize raw market data."""
        frame = raw_df.copy()
        frame["ds"] = pd.to_datetime(frame["ds"])
        frame["unique_id"] = frame["unique_id"].astype(str).str.upper()
        numeric_cols = ["open", "high", "low", "close", "volume"]
        frame[numeric_cols] = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=["ds", "open", "high", "low", "close", "volume"])
        frame = frame.loc[(frame["close"] > 0) & (frame["volume"] > 0)].copy()
        frame = frame.drop_duplicates(subset=["unique_id", "ds"]).sort_values(["unique_id", "ds"])
        frame = self._regularize_frequency(frame)

        # Compute target based on target_type
        frame = self._compute_target(frame)

        frame["sector"] = frame["unique_id"].map(self.sector_map).fillna("Unknown")
        frame["asset_class"] = self.asset_class
        sector_codes = {name: idx + 1 for idx, name in enumerate(sorted(frame["sector"].unique()))}
        frame["sector_code"] = frame["sector"].map(sector_codes).astype(int)
        frame["asset_class_code"] = 1
        frame["volume"] = frame["volume"].round().astype(int)
        max_volume = frame.groupby("unique_id")["volume"].transform("max").replace(0, 1)
        frame["sample_weight"] = (frame["volume"] / max_volume).clip(lower=0.05)
        return frame.reset_index(drop=True)

    def _regularize_frequency(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps in market calendar."""
        parts: list[pd.DataFrame] = []
        for unique_id, grp in frame.groupby("unique_id", sort=True):
            grp = grp.sort_values("ds").set_index("ds")
            full_dates = pd.date_range(grp.index.min(), grp.index.max(), freq=self.freq)
            reindexed = grp.reindex(full_dates)
            reindexed["unique_id"] = unique_id
            for col in ["open", "high", "low", "close"]:
                reindexed[col] = reindexed[col].ffill().bfill()
            reindexed["volume"] = (
                reindexed["volume"].ffill().bfill().fillna(1).clip(lower=1).round()
            )
            reindexed = reindexed.reset_index().rename(columns={"index": "ds"})
            parts.append(reindexed[["unique_id", "ds", "open", "high", "low", "close", "volume"]])
        return pd.concat(parts, ignore_index=True)

    def add_calendar_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features."""
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

    def add_technical_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features per series."""
        result = frame.copy()

        for uid in result["unique_id"].unique():
            mask = result["unique_id"] == uid
            close = result.loc[mask, "close"]
            high = result.loc[mask, "high"]
            low = result.loc[mask, "low"]
            volume = result.loc[mask, "volume"]

            # Volatility (rolling std of returns)
            returns = np.log(close / close.shift(1))
            result.loc[mask, "volatility_5d"] = returns.rolling(5).std().fillna(0)
            result.loc[mask, "volatility_20d"] = returns.rolling(20).std().fillna(0)

            # Price momentum
            result.loc[mask, "momentum_5d"] = (close / close.shift(5) - 1).fillna(0)
            result.loc[mask, "momentum_20d"] = (close / close.shift(20) - 1).fillna(0)

            # Range (high-low) normalized
            result.loc[mask, "range_pct"] = ((high - low) / close).fillna(0)

            # Volume momentum
            result.loc[mask, "volume_ma_ratio"] = (
                volume / volume.rolling(20).mean()
            ).fillna(1).clip(0.1, 10)

        return result

    def build_training_frame(
        self, raw_df: pd.DataFrame, validate_rows: bool = True
    ) -> pd.DataFrame:
        """Build complete training frame with all features."""
        normalized = self.normalize_market_frame(raw_df)
        with_calendar = self.add_calendar_features(normalized)
        training_frame = self.add_technical_features(with_calendar)
        if validate_rows:
            validate_market_rows(training_frame)
        return training_frame

    def build_static_features(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        """Extract static features for each series."""
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
        """Build future exogenous features for forecasting."""
        future_rows: list[pd.DataFrame] = []
        for unique_id in ids:
            horizon_dates = pd.date_range(last_timestamp, periods=horizon + 1, freq=freq)[1:]
            frame = pd.DataFrame({"unique_id": unique_id, "ds": horizon_dates})
            future_rows.append(frame)
        future_df = pd.concat(future_rows, ignore_index=True)
        return self.add_calendar_features(future_df)

    def get_last_prices(self, training_frame: pd.DataFrame) -> pd.DataFrame:
        """Get last known price for each series."""
        last_prices = (
            training_frame.groupby("unique_id")
            .agg({"close": "last", "ds": "max"})
            .reset_index()
            .rename(columns={"close": "last_price", "ds": "last_ds"})
        )
        return last_prices

    def quality_report(self, frame: pd.DataFrame) -> DataQualityReport:
        """Generate quality report for the data."""
        return DataQualityReport(
            rows=int(len(frame)),
            series=int(frame["unique_id"].nunique()),
            start=pd.Timestamp(frame["ds"].min()),
            end=pd.Timestamp(frame["ds"].max()),
            missing_rate=float(frame.isna().mean().mean()),
            target_type=self.target_type.value,
        )

    def holdout_split(
        self, frame: pd.DataFrame, horizon: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
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
