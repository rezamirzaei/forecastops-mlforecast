from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from mlforecast_realworld.config import AppSettings
from mlforecast_realworld.ml import factory as factory_mod
from mlforecast_realworld.ml import pipeline as pipeline_mod
from mlforecast_realworld.ml.pipeline import (
    ForecastPipeline,
    after_predict_clip,
    before_predict_cleanup,
)


def test_before_predict_cleanup() -> None:
    frame = pd.DataFrame({"x": [1.0, None], "name": ["a", "b"]})
    cleaned = before_predict_cleanup(frame)
    assert cleaned["x"].isna().sum() == 0


def test_after_predict_clip_series() -> None:
    series = pd.Series([-1.0, 2.0])
    clipped = after_predict_clip(series)
    assert float(clipped.min()) >= 0


def _test_settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        environment="test",
        data={"tickers": ["aapl.us", "msft.us"]},
        paths={
            "project_root": str(tmp_path),
            "raw_data_dir": "raw",
            "processed_data_dir": "processed",
            "model_dir": "models",
            "report_dir": "reports",
        },
        forecast={
            "freq": "B",
            "horizon": 3,
            "lags": [1, 2, 3],
            "cv_windows": 2,
            "cv_step_size": 2,
            "levels": [80],
            "keep_last_n": 120,
            "num_threads": 1,
            "target_type": "price",  # Use price for test backward compatibility
        },
    )


def test_pipeline_full_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_raw_frame: pd.DataFrame,
) -> None:
    settings = _test_settings(tmp_path)
    pipeline = ForecastPipeline(settings=settings)

    monkeypatch.setattr(pipeline.downloader, "download_all", lambda: sample_raw_frame)
    monkeypatch.setattr(
        factory_mod,
        "default_models",
        lambda random_state: {"lin_reg": LinearRegression()},
    )

    frame = pipeline.prepare_training_data(download=True)
    assert pipeline.raw_data_path.exists()
    assert pipeline.processed_data_path.exists()

    fitted = pipeline.fit(frame)
    assert not fitted.empty

    cv_df, cv_summary = pipeline.cross_validate(frame)
    assert not cv_df.empty
    assert not cv_summary.empty

    preds = pipeline.forecast(horizon=3, ids=["AAPL.US", "MSFT.US"])
    assert len(preds) == 6

    model_path = pipeline.save_model()
    assert model_path.exists()


def test_update_requires_fit(tmp_path: Path) -> None:
    pipeline = ForecastPipeline(settings=_test_settings(tmp_path))
    with pytest.raises(RuntimeError):
        pipeline.update_with_latest(pd.DataFrame())


def test_cross_validate_fallbacks_when_intervals_need_missing_xdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sample_raw_frame: pd.DataFrame,
) -> None:
    settings = _test_settings(tmp_path)
    settings.forecast.enable_prediction_intervals = True
    pipeline = ForecastPipeline(settings=settings)
    frame = pipeline.data_engineer.build_training_frame(sample_raw_frame)
    pipeline.training_frame = frame
    pipeline.forecaster = object()

    class StubCVForecaster:
        def __init__(self) -> None:
            self.calls = 0
            self.prediction_intervals: list[object | None] = []
            self.models = {"lin_reg": object(), "rf": object()}

        def cross_validation(self, *_args, **kwargs):
            self.calls += 1
            self.prediction_intervals.append(kwargs.get("prediction_intervals"))
            if self.calls == 1:
                raise ValueError("Found missing inputs in X_df. It should have one row per id.")
            return pd.DataFrame(
                {
                    "unique_id": ["AAPL.US", "MSFT.US"],
                    "ds": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
                    "cutoff": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
                    "y": [100.0, 200.0],
                    "lin_reg": [99.0, 198.0],
                    "rf": [100.0, 199.0],
                }
            )

        @staticmethod
        def cross_validation_fitted_values() -> pd.DataFrame:
            return pd.DataFrame()

    stub = StubCVForecaster()
    monkeypatch.setattr(pipeline_mod, "build_mlforecast", lambda _cfg: stub)

    cv_df, summary = pipeline.cross_validate(frame)

    assert stub.calls == 2
    assert stub.prediction_intervals[0] is not None
    assert stub.prediction_intervals[1] is None
    assert pipeline.intervals_enabled is False
    assert not cv_df.empty
    assert not summary.empty


def test_forecast_filters_ids_without_passing_subset_to_predict(
    tmp_path: Path,
    sample_raw_frame: pd.DataFrame,
) -> None:
    pipeline = ForecastPipeline(settings=_test_settings(tmp_path))
    frame = pipeline.data_engineer.build_training_frame(sample_raw_frame)
    pipeline.training_frame = frame

    class StubForecaster:
        models = {"lin_reg": object(), "rf": object()}

        @staticmethod
        def get_missing_future(h: int, X_df: pd.DataFrame) -> pd.DataFrame:
            for col in [
                "volatility_5d",
                "volatility_20d",
                "momentum_5d",
                "momentum_20d",
                "range_pct",
                "volume_ma_ratio",
            ]:
                assert col in X_df.columns
            return pd.DataFrame(columns=X_df.columns)

        @staticmethod
        def make_future_dataframe(h: int) -> pd.DataFrame:
            return pd.DataFrame({"h": [h]})

        @staticmethod
        def predict(*_args, **kwargs) -> pd.DataFrame:
            assert kwargs.get("ids") is None
            return pd.DataFrame(
                {
                    "unique_id": ["AAPL.US", "MSFT.US"],
                    "ds": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
                    "lin_reg": [100.0, 200.0],
                    "rf": [110.0, 210.0],
                }
            )

    pipeline.forecaster = StubForecaster()  # type: ignore[assignment]

    preds = pipeline.forecast(horizon=1, ids=["AAPL.US"])

    assert preds["unique_id"].tolist() == ["AAPL.US"]
    assert "ensemble_mean" in preds.columns
    assert preds["ensemble_mean"].tolist() == [105.0]


def test_attach_latest_technical_features_carries_last_observation(
    tmp_path: Path,
) -> None:
    pipeline = ForecastPipeline(settings=_test_settings(tmp_path))
    future_x = pd.DataFrame(
        {
            "unique_id": ["AAPL.US", "AAPL.US", "MSFT.US"],
            "ds": pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-03"]),
        }
    )
    training = pd.DataFrame(
        {
            "unique_id": ["AAPL.US", "AAPL.US", "MSFT.US"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-02"]),
            "volatility_5d": [0.1, 0.2, 0.3],
            "volatility_20d": [0.4, 0.5, 0.6],
            "momentum_5d": [0.01, 0.02, 0.03],
            "momentum_20d": [0.04, 0.05, 0.06],
            "range_pct": [0.07, 0.08, 0.09],
            "volume_ma_ratio": [1.1, 1.2, 1.3],
        }
    )

    out = pipeline._attach_latest_technical_features(future_x, training)
    aapl_rows = out[out["unique_id"] == "AAPL.US"]
    assert set(aapl_rows["volatility_5d"].tolist()) == {0.2}
    assert set(aapl_rows["momentum_20d"].tolist()) == {0.05}
    assert set(aapl_rows["volume_ma_ratio"].tolist()) == {1.2}


def test_get_fitted_values_computes_runtime_from_model_weights(tmp_path: Path) -> None:
    pipeline = ForecastPipeline(settings=_test_settings(tmp_path))
    pipeline.training_frame = pd.DataFrame(
        {
            "unique_id": ["AAPL.US", "AAPL.US", "MSFT.US"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "y": [0.1, 0.2, 0.3],
        }
    )

    class StubModel:
        @staticmethod
        def predict(X):
            return np.full(len(X), 0.42)

    class StubForecaster:
        models = {"stub_model": StubModel()}
        models_ = {"stub_model": StubModel()}

        @staticmethod
        def forecast_fitted_values():
            raise RuntimeError("not available")

        @staticmethod
        def preprocess(frame, static_features=None, keep_last_n=None, as_numpy=False):  # noqa: ARG002
            out = frame.copy()
            out["lag1"] = 1.0
            return out

    pipeline.forecaster = StubForecaster()  # type: ignore[assignment]
    fitted = pipeline.get_fitted_values(ids=["AAPL.US"])
    assert not fitted.empty
    assert set(fitted["unique_id"].tolist()) == {"AAPL.US"}
    assert "stub_model" in fitted.columns
    assert set(fitted["stub_model"].tolist()) == {0.42}


def test_forecast_injects_unseen_ids_history_into_forecaster(tmp_path: Path) -> None:
    pipeline = ForecastPipeline(settings=_test_settings(tmp_path))
    pipeline.training_frame = pd.DataFrame(
        {
            "unique_id": ["AAPL.US", "AAPL.US", "MSFT.US", "MSFT.US"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "y": [0.1, 0.2, 0.3, 0.4],
            "sector_code": [1, 1, 1, 1],
            "asset_class_code": [1, 1, 1, 1],
            "is_weekend": [0, 0, 0, 0],
            "is_month_start": [1, 0, 1, 0],
            "is_month_end": [0, 0, 0, 0],
            "week_of_year": [1, 1, 1, 1],
            "month_sin": [0.0, 0.0, 0.0, 0.0],
            "month_cos": [1.0, 1.0, 1.0, 1.0],
        }
    )

    class StubForecaster:
        def __init__(self) -> None:
            self.models = {"lin_reg": object()}
            self.ts = type("TS", (), {"uids": pd.Index(["AAPL.US"], name="unique_id")})()
            self.updated_ids: list[str] = []

        @staticmethod
        def get_missing_future(h: int, X_df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(columns=X_df.columns)

        @staticmethod
        def make_future_dataframe(h: int) -> pd.DataFrame:
            return pd.DataFrame({"h": [h]})

        def update(self, new_observations: pd.DataFrame) -> None:
            self.updated_ids = sorted(new_observations["unique_id"].unique().tolist())
            self.ts.uids = pd.Index(["AAPL.US", "MSFT.US"], name="unique_id")

        @staticmethod
        def predict(*_args, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "unique_id": ["AAPL.US", "MSFT.US"],
                    "ds": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
                    "lin_reg": [0.11, 0.33],
                }
            )

    forecaster = StubForecaster()
    pipeline.forecaster = forecaster  # type: ignore[assignment]

    preds = pipeline.forecast(horizon=1, ids=["MSFT.US"], return_prices=False)
    assert forecaster.updated_ids == ["MSFT.US"]
    assert preds["unique_id"].tolist() == ["MSFT.US"]


def test_forecast_injects_unseen_ids_via_preprocess_fallback(tmp_path: Path) -> None:
    """When update() raises ValueError (target_transforms), we fall back to preprocess."""
    pipeline = ForecastPipeline(settings=_test_settings(tmp_path))
    pipeline.training_frame = pd.DataFrame(
        {
            "unique_id": ["AAPL.US", "AAPL.US", "MSFT.US", "MSFT.US"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
            "y": [0.1, 0.2, 0.3, 0.4],
            "sector_code": [1, 1, 1, 1],
            "asset_class_code": [1, 1, 1, 1],
            "is_weekend": [0, 0, 0, 0],
            "is_month_start": [1, 0, 1, 0],
            "is_month_end": [0, 0, 0, 0],
            "week_of_year": [1, 1, 1, 1],
            "month_sin": [0.0, 0.0, 0.0, 0.0],
            "month_cos": [1.0, 1.0, 1.0, 1.0],
        }
    )

    class StubForecaster:
        def __init__(self) -> None:
            self.models = {"lin_reg": object()}
            self.ts = type("TS", (), {"uids": pd.Index(["AAPL.US"], name="unique_id")})()
            self.preprocessed = False

        @staticmethod
        def get_missing_future(h: int, X_df: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame(columns=X_df.columns)

        @staticmethod
        def make_future_dataframe(h: int) -> pd.DataFrame:
            return pd.DataFrame({"h": [h]})

        def update(self, new_observations: pd.DataFrame) -> None:
            raise ValueError("Can not update target_transforms with new series.")

        def preprocess(self, frame, **_kwargs) -> pd.DataFrame:
            self.preprocessed = True
            self.ts.uids = pd.Index(
                sorted(frame["unique_id"].unique()), name="unique_id"
            )
            return frame

        @staticmethod
        def predict(*_args, **_kwargs) -> pd.DataFrame:
            return pd.DataFrame(
                {
                    "unique_id": ["AAPL.US", "MSFT.US"],
                    "ds": [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-03")],
                    "lin_reg": [0.11, 0.33],
                }
            )

    forecaster = StubForecaster()
    pipeline.forecaster = forecaster  # type: ignore[assignment]

    preds = pipeline.forecast(horizon=1, ids=["MSFT.US"], return_prices=False)
    assert forecaster.preprocessed, "Should have fallen back to preprocess"
    assert preds["unique_id"].tolist() == ["MSFT.US"]


def test_load_model_retries_with_numpy_pickle_aliases(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pipeline = ForecastPipeline(settings=_test_settings(tmp_path))
    sentinel = object()
    state = {"first": True}

    def fake_load(_path):
        if state["first"]:
            state["first"] = False
            raise ModuleNotFoundError("No module named 'numpy._core.numeric'")
        return sentinel

    monkeypatch.setattr(pipeline_mod.MLForecast, "load", staticmethod(fake_load))

    loaded = pipeline.load_model()
    assert loaded is sentinel
