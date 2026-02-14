from __future__ import annotations

from pathlib import Path

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
        @staticmethod
        def get_missing_future(h: int, X_df: pd.DataFrame) -> pd.DataFrame:
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
                }
            )

    pipeline.forecaster = StubForecaster()  # type: ignore[assignment]

    preds = pipeline.forecast(horizon=1, ids=["AAPL.US"])

    assert preds["unique_id"].tolist() == ["AAPL.US"]
