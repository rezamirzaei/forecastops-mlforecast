from mlforecast import MLForecast

from mlforecast_realworld.config import ForecastSettings
from mlforecast_realworld.ml.factory import build_mlforecast, default_models, lag_transform_namer


class DummyTransform:
    pass


def test_lag_transform_namer_with_class_instance() -> None:
    name = lag_transform_namer(DummyTransform(), 7)
    assert name == "dummytransform_lag7"


def test_lag_transform_namer_with_args() -> None:
    name = lag_transform_namer(DummyTransform(), 7, 3)
    assert name == "dummytransform_lag7_3"


def test_default_models_returns_expected_keys() -> None:
    models = default_models(42)
    assert {"lin_reg", "rf", "lgbm", "xgb"}.issubset(models)


def test_build_mlforecast_returns_instance() -> None:
    forecast = build_mlforecast(ForecastSettings())
    assert isinstance(forecast, MLForecast)
    assert forecast.freq == "B"
