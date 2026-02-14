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


def week_of_month(dates):
    if hasattr(dates, "dt"):
        day = dates.dt.day
    else:
        day = pd.Index(dates).day
    return ((day - 1) // 7) + 1


def lag_transform_namer(tfm: Any, lag: int, *args: Any) -> str:
    base = tfm.__name__ if hasattr(tfm, "__name__") else tfm.__class__.__name__.lower()
    if args:
        args_part = "_".join(str(arg) for arg in args)
        return f"{base}_lag{lag}_{args_part}"
    return f"{base}_lag{lag}"


def default_models(random_state: int) -> dict[str, Any]:
    return {
        "lin_reg": LinearRegression(),
        "enet": ElasticNet(alpha=0.001, l1_ratio=0.15, max_iter=5000, random_state=random_state),
        "rf": RandomForestRegressor(
            n_estimators=120,
            random_state=random_state,
            n_jobs=-1,
            max_depth=8,
            min_samples_leaf=2,
        ),
        "etr": ExtraTreesRegressor(
            n_estimators=180,
            random_state=random_state,
            n_jobs=-1,
            max_depth=10,
            min_samples_leaf=2,
        ),
        "hgb": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=220,
            max_depth=8,
            min_samples_leaf=20,
            random_state=random_state,
        ),
        "lgbm": LGBMRegressor(
            n_estimators=160,
            learning_rate=0.05,
            random_state=random_state,
            objective="regression",
            n_jobs=-1,
            verbose=-1,
            force_col_wise=True,
        ),
        "xgb": XGBRegressor(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=2,
            objective="reg:squarederror",
            verbosity=0,
        ),
    }


def build_mlforecast(settings: ForecastSettings) -> MLForecast:
    season_length = int(settings.season_length)
    lag_transforms = {
        1: [ExpandingMean(), ExpandingStd()],
        7: [RollingMean(window_size=7), RollingStd(window_size=7)],
        14: [SeasonalRollingMean(season_length=season_length, window_size=2)],
        21: [ExponentiallyWeightedMean(alpha=0.3)],
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
