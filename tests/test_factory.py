from mlforecast import MLForecast

from mlforecast_realworld.config import ForecastSettings
from mlforecast_realworld.ml.factory import (
    ModelComplexity,
    ModelRegistry,
    build_mlforecast,
    default_models,
    get_models_for_complexity,
    lag_transform_namer,
)


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
    # New model set includes ridge, enet, rf, etr, gb, hgb, lgbm, xgb, mlp
    assert {"ridge", "rf", "lgbm", "xgb", "mlp"}.issubset(models)


def test_build_mlforecast_returns_instance() -> None:
    forecast = build_mlforecast(ForecastSettings())
    assert isinstance(forecast, MLForecast)
    assert forecast.freq == "B"


def test_model_registry_get_all_models() -> None:
    models = ModelRegistry.get_all_models(42)
    assert len(models) >= 9  # At least 9 models
    assert "ridge" in models
    assert "mlp" in models


def test_model_registry_get_models_by_complexity() -> None:
    simple = ModelRegistry.get_models_by_complexity(ModelComplexity.SIMPLE, 42)
    assert "ridge" in simple
    assert "enet" in simple

    complex_models = ModelRegistry.get_models_by_complexity(ModelComplexity.COMPLEX, 42)
    assert "lgbm" in complex_models
    assert "mlp" in complex_models


def test_model_registry_list_models() -> None:
    model_list = ModelRegistry.list_models()
    assert len(model_list) >= 9
    assert all("name" in m and "complexity" in m and "description" in m for m in model_list)


def test_get_models_for_complexity() -> None:
    moderate = get_models_for_complexity(ModelComplexity.MODERATE, 42)
    assert "rf" in moderate
    assert "etr" in moderate
