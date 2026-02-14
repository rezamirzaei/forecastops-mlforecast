"""
MLForecast model factory.

This module provides factory functions to construct MLForecast instances
with pre-configured models, lag transforms, and date features suitable
for financial time-series forecasting.

Key components:
- default_models(): Returns a dictionary of sklearn-compatible regressors
- build_mlforecast(): Constructs an MLForecast instance with full configuration
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from lightgbm import LGBMRegressor
from mlforecast import MLForecast
from mlforecast.lag_transforms import (
    ExpandingMean,
    ExpandingStd,
    ExponentiallyWeightedMean,
    RollingMean,
    RollingStd,
    SeasonalRollingMean,
)
from mlforecast.target_transforms import Differences, LocalStandardScaler
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, LinearRegression
from xgboost import XGBRegressor

from mlforecast_realworld.config import ForecastSettings

# ============================================================================
# Constants
# ============================================================================

# Model hyperparameters
DEFAULT_N_ESTIMATORS_RF = 120
DEFAULT_N_ESTIMATORS_ETR = 180
DEFAULT_N_ESTIMATORS_HGB = 220
DEFAULT_N_ESTIMATORS_LGBM = 160
DEFAULT_N_ESTIMATORS_XGB = 180
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_MAX_DEPTH_TREE = 8
DEFAULT_MAX_DEPTH_XGB = 5
DEFAULT_MIN_SAMPLES_LEAF = 2
DEFAULT_MIN_SAMPLES_LEAF_HGB = 20

# Lag transform window sizes
ROLLING_WINDOW_7 = 7
ROLLING_WINDOW_14 = 14
ROLLING_WINDOW_21 = 21
EWM_ALPHA = 0.3


# ============================================================================
# Date Feature Functions
# ============================================================================

def week_of_month(dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
    """
    Calculate the week of month (1-5) for given dates.

    Args:
        dates: DatetimeIndex or Series with datetime values.

    Returns:
        Series with week of month values (1 = first week, etc.).
    """
    if hasattr(dates, "dt"):
        day = dates.dt.day
    else:
        day = pd.Index(dates).day
    return ((day - 1) // 7) + 1


# ============================================================================
# Transform Naming
# ============================================================================

def lag_transform_namer(tfm: Any, lag: int, *args: Any) -> str:
    """
    Generate consistent feature names for lag transforms.

    Args:
        tfm: The transform class or instance.
        lag: The lag value being transformed.
        *args: Additional arguments passed to the transform.

    Returns:
        A descriptive feature name like 'rollingmean_lag7_7'.
    """
    base = tfm.__name__ if hasattr(tfm, "__name__") else tfm.__class__.__name__.lower()
    if args:
        args_part = "_".join(str(arg) for arg in args)
        return f"{base}_lag{lag}_{args_part}"
    return f"{base}_lag{lag}"


# ============================================================================
# Model Factory
# ============================================================================

def default_models(random_state: int) -> dict[str, Any]:
    """
    Create the default ensemble of regression models for forecasting.

    This ensemble covers a range of model families:
    - Linear models: LinearRegression, ElasticNet
    - Tree ensembles: RandomForest, ExtraTrees
    - Gradient boosting: HistGradientBoosting, LightGBM, XGBoost

    Args:
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary mapping model names to fitted sklearn-compatible estimators.
    """
    return {
        "lin_reg": LinearRegression(),
        "enet": ElasticNet(
            alpha=0.001,
            l1_ratio=0.15,
            max_iter=5000,
            random_state=random_state,
        ),
        "rf": RandomForestRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS_RF,
            random_state=random_state,
            n_jobs=-1,
            max_depth=DEFAULT_MAX_DEPTH_TREE,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        ),
        "etr": ExtraTreesRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS_ETR,
            random_state=random_state,
            n_jobs=-1,
            max_depth=DEFAULT_MAX_DEPTH_TREE + 2,  # Slightly deeper for ETR
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF,
        ),
        "hgb": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=DEFAULT_LEARNING_RATE,
            max_iter=DEFAULT_N_ESTIMATORS_HGB,
            max_depth=DEFAULT_MAX_DEPTH_TREE,
            min_samples_leaf=DEFAULT_MIN_SAMPLES_LEAF_HGB,
            random_state=random_state,
        ),
        "lgbm": LGBMRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS_LGBM,
            learning_rate=DEFAULT_LEARNING_RATE,
            random_state=random_state,
            objective="regression",
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
        ),
        "xgb": XGBRegressor(
            n_estimators=DEFAULT_N_ESTIMATORS_XGB,
            learning_rate=DEFAULT_LEARNING_RATE,
            max_depth=DEFAULT_MAX_DEPTH_XGB,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=2,
            objective="reg:squarederror",
            verbosity=0,
        ),
    }


def build_mlforecast(settings: ForecastSettings) -> MLForecast:
    """
    Build a fully configured MLForecast instance from settings.

    This factory configures:
    - Multiple regression models via default_models()
    - Lag transforms: rolling stats, expanding stats, EWM, seasonal
    - Date features: dayofweek, month, quarter, week_of_month
    - Target transforms: differencing and local standardization

    Args:
        settings: ForecastSettings with horizon, lags, frequency, etc.

    Returns:
        Configured MLForecast instance ready for fit/predict.
    """
    season_length = int(settings.season_length)

    lag_transforms = {
        1: [ExpandingMean(), ExpandingStd()],
        ROLLING_WINDOW_7: [
            RollingMean(window_size=ROLLING_WINDOW_7),
            RollingStd(window_size=ROLLING_WINDOW_7),
        ],
        ROLLING_WINDOW_14: [
            SeasonalRollingMean(season_length=season_length, window_size=2),
        ],
        ROLLING_WINDOW_21: [
            ExponentiallyWeightedMean(alpha=EWM_ALPHA),
        ],
    }

    date_features = ["dayofweek", "month", "quarter", week_of_month]
    target_transforms = [Differences([1]), LocalStandardScaler()]

    return MLForecast(
        models=default_models(settings.random_state),
        freq=settings.freq,
        lags=settings.lags,
        lag_transforms=lag_transforms,
        date_features=date_features,
        num_threads=settings.num_threads,
        target_transforms=target_transforms,
        lag_transforms_namer=lag_transform_namer,
    )
