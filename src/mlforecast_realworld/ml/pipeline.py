"""
Forecasting pipeline orchestration.

This module provides the ForecastPipeline class which orchestrates the end-to-end
ML forecasting workflow:
- Data download and preparation
- Model training with MLForecast
- Cross-validation and evaluation
- Prediction and model persistence

Example:
    >>> pipeline = ForecastPipeline()
    >>> pipeline.prepare_training_data(download=True)
    >>> pipeline.fit()
    >>> predictions = pipeline.predict(horizon=14, ids=["AAPL.US"])
"""
from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lgb_cv import LightGBMCV
from mlforecast.utils import PredictionIntervals

from mlforecast_realworld.config import AppSettings, get_settings
from mlforecast_realworld.data.downloader import StooqDownloader
from mlforecast_realworld.data.engineering import MarketDataEngineer, TargetType
from mlforecast_realworld.ml.evaluation import summarize_cv
from mlforecast_realworld.ml.factory import build_mlforecast
from mlforecast_realworld.utils.io import ensure_directory, load_parquet, save_json, save_parquet


def before_predict_cleanup(features: np.ndarray | pd.DataFrame) -> np.ndarray | pd.DataFrame:
    """
    Clean up feature matrix before prediction by replacing NaN values.

    Args:
        features: Feature matrix as NumPy array or DataFrame.

    Returns:
        Cleaned feature matrix with NaN replaced by 0.0.
    """
    if isinstance(features, np.ndarray):
        return np.nan_to_num(features, nan=0.0)
    cleaned = features.copy()
    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    cleaned[numeric_cols] = cleaned[numeric_cols].fillna(0.0)
    return cleaned


def after_predict_clip(values: pd.Series | pd.DataFrame | np.ndarray) -> Any:
    """
    Clip prediction values to ensure non-negative outputs.

    Args:
        values: Prediction values to clip.

    Returns:
        Values clipped to minimum of 0.
    """
    if isinstance(values, pd.Series):
        return values.clip(lower=0)
    if isinstance(values, pd.DataFrame):
        return values.clip(lower=0)
    if isinstance(values, np.ndarray):
        return np.clip(values, a_min=0, a_max=None)
    return values


class ForecastPipeline:
    """
    End-to-end ML forecasting pipeline using MLForecast.

    This class orchestrates the complete forecasting workflow including data
    preparation, model training, cross-validation, prediction, and persistence.

    Attributes:
        settings: Application settings containing paths and model configuration.
        forecaster: Trained MLForecast instance (None until fit is called).
        training_frame: Prepared training DataFrame.
        intervals_enabled: Whether prediction intervals are available.

    Example:
        >>> pipeline = ForecastPipeline()
        >>> pipeline.prepare_training_data(download=True)
        >>> fitted_values = pipeline.fit()
        >>> predictions = pipeline.predict(horizon=14, ids=["AAPL.US"])
    """

    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        # Initialize data engineer with target type from settings
        target_type = TargetType(self.settings.forecast.target_type)
        self.data_engineer = MarketDataEngineer(target_type=target_type)
        self.downloader = StooqDownloader(
            settings=self.settings.data,
            output_dir=self.settings.resolved_path(self.settings.paths.raw_data_dir),
        )
        self.forecaster: MLForecast | None = None
        self.manual_forecaster: MLForecast | None = None
        self.training_frame: pd.DataFrame | None = None
        self.latest_cv_summary: pd.DataFrame | None = None
        self.intervals_enabled = False
        self._last_prices: pd.DataFrame | None = None
        self._ensure_dirs()

    @property
    def raw_data_path(self) -> Path:
        return self.settings.resolved_path(self.settings.paths.raw_data_dir) / "market_raw.parquet"

    @property
    def processed_data_path(self) -> Path:
        resolved = self.settings.resolved_path(self.settings.paths.processed_data_dir)
        return resolved / "market_training.parquet"

    @property
    def model_path(self) -> Path:
        return self.settings.resolved_path(self.settings.paths.model_dir) / "mlforecast_model"

    @property
    def fitted_values_path(self) -> Path:
        return self.settings.resolved_path(self.settings.paths.model_dir) / "fitted_values.parquet"

    @property
    def report_path(self) -> Path:
        return self.settings.resolved_path(self.settings.paths.report_dir) / "pipeline_report.json"

    @property
    def run_summary_path(self) -> Path:
        return self.settings.resolved_path(self.settings.paths.report_dir) / "run_summary.json"

    @property
    def cv_summary_path(self) -> Path:
        return self.settings.resolved_path(self.settings.paths.report_dir) / "cv_summary.parquet"

    def _ensure_dirs(self) -> None:
        for directory in self.settings.paths.all_dirs():
            ensure_directory(self.settings.resolved_path(directory))

    def prepare_training_data(self, download: bool = True) -> pd.DataFrame:
        if download or not self.raw_data_path.exists():
            raw = self.downloader.download_all()
            self.downloader.save_raw(raw)
        else:
            raw = load_parquet(self.raw_data_path)

        training = self.data_engineer.build_training_frame(raw)
        save_parquet(training, self.processed_data_path)
        report = asdict(self.data_engineer.quality_report(training))
        save_json(report, self.report_path)
        self.training_frame = training
        # Store last prices for price reconstruction from returns
        self._last_prices = self.data_engineer.get_last_prices(training)
        return training

    def _require_training_frame(self) -> pd.DataFrame:
        if self.training_frame is not None:
            return self.training_frame
        if self.processed_data_path.exists():
            self.training_frame = load_parquet(self.processed_data_path)
            return self.training_frame
        raise RuntimeError("training data not prepared")

    def fit(self, training_frame: pd.DataFrame | None = None) -> pd.DataFrame:
        frame = training_frame if training_frame is not None else self._require_training_frame()
        model_frame = self._model_frame(frame)
        static_cols = ["sector_code", "asset_class_code"]

        forecaster = build_mlforecast(self.settings.forecast)

        # Feature matrix inspection for data engineering/debugging.
        forecaster.preprocess(
            model_frame,
            static_features=static_cols,
            keep_last_n=self.settings.forecast.keep_last_n,
        )
        X, y = forecaster.preprocess(
            model_frame,
            static_features=static_cols,
            keep_last_n=self.settings.forecast.keep_last_n,
            return_X_y=True,
            as_numpy=True,
        )

        manual_forecaster = build_mlforecast(self.settings.forecast)
        manual_forecaster.preprocess(
            model_frame,
            static_features=static_cols,
            keep_last_n=self.settings.forecast.keep_last_n,
            as_numpy=True,
        )
        manual_forecaster.fit_models(X, y)
        self.manual_forecaster = manual_forecaster

        use_intervals = self.settings.forecast.enable_prediction_intervals
        intervals = (
            PredictionIntervals(
                n_windows=2,
                h=self.settings.forecast.horizon,
                method="conformal_distribution",
            )
            if use_intervals
            else None
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, but LGBMRegressor was fitted with "
                    "feature names"
                ),
            )
            if intervals is None:
                forecaster.fit(
                    model_frame,
                    static_features=static_cols,
                    keep_last_n=self.settings.forecast.keep_last_n,
                    fitted=True,
                    as_numpy=True,
                )
                self.intervals_enabled = False
            else:
                try:
                    forecaster.fit(
                        model_frame,
                        static_features=static_cols,
                        keep_last_n=self.settings.forecast.keep_last_n,
                        prediction_intervals=intervals,
                        fitted=True,
                        as_numpy=True,
                    )
                    self.intervals_enabled = True
                except ValueError as exc:
                    msg = str(exc)
                    if "missing inputs in X_df" not in msg:
                        raise
                    # Real-world market calendars have holidays/gaps. Fallback keeps API responsive.
                    forecaster.fit(
                        model_frame,
                        static_features=static_cols,
                        keep_last_n=self.settings.forecast.keep_last_n,
                        fitted=True,
                        as_numpy=True,
                    )
                    self.intervals_enabled = False
        self.forecaster = forecaster
        fitted = forecaster.forecast_fitted_values()
        save_parquet(fitted, self.fitted_values_path)
        return fitted

    def cross_validate(
        self, training_frame: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        frame = training_frame if training_frame is not None else self._require_training_frame()
        model_frame = self._model_frame(frame)
        if self.forecaster is None:
            self.fit(frame)

        use_intervals = self.settings.forecast.enable_prediction_intervals
        intervals = (
            PredictionIntervals(
                n_windows=2,
                h=self.settings.forecast.horizon,
                method="conformal_distribution",
            )
            if use_intervals
            else None
        )
        cv_forecaster = build_mlforecast(self.settings.forecast)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, but LGBMRegressor was fitted with "
                    "feature names"
                ),
            )
            try:
                levels: list[int | float] | None = (
                    list(self.settings.forecast.levels) if intervals is not None else None
                )
                cv_df = cv_forecaster.cross_validation(
                    model_frame,
                    n_windows=self.settings.forecast.cv_windows,
                    h=self.settings.forecast.horizon,
                    step_size=self.settings.forecast.cv_step_size,
                    static_features=["sector_code", "asset_class_code"],
                    keep_last_n=self.settings.forecast.keep_last_n,
                    refit=1,
                    fitted=True,
                    prediction_intervals=intervals,
                    level=levels,
                    before_predict_callback=before_predict_cleanup,
                    after_predict_callback=after_predict_clip,
                    as_numpy=True,
                )
            except ValueError as exc:
                msg = str(exc)
                if intervals is None or "missing inputs in X_df" not in msg:
                    raise
                # Some CV windows still hit sparse market calendars; rerun CV without intervals.
                cv_df = cv_forecaster.cross_validation(
                    model_frame,
                    n_windows=self.settings.forecast.cv_windows,
                    h=self.settings.forecast.horizon,
                    step_size=self.settings.forecast.cv_step_size,
                    static_features=["sector_code", "asset_class_code"],
                    keep_last_n=self.settings.forecast.keep_last_n,
                    refit=1,
                    fitted=True,
                    prediction_intervals=None,
                    level=None,
                    before_predict_callback=before_predict_cleanup,
                    after_predict_callback=after_predict_clip,
                    as_numpy=True,
                )
                self.intervals_enabled = False
        cv_df = self._add_ensemble_column(cv_df, model_names=list(cv_forecaster.models.keys()))
        _ = cv_forecaster.cross_validation_fitted_values()
        summary = summarize_cv(cv_df)
        self.latest_cv_summary = summary
        return cv_df, summary

    def forecast(
        self,
        horizon: int | None = None,
        ids: list[str] | None = None,
        levels: list[int] | None = None,
        return_prices: bool = True,
    ) -> pd.DataFrame:
        """
        Generate forecasts for specified series.

        Args:
            horizon: Forecast horizon (number of steps).
            ids: List of series IDs to forecast.
            levels: Confidence levels for prediction intervals.
            return_prices: If True and target is returns, reconstruct prices.

        Returns:
            DataFrame with predictions (prices or returns based on settings).
        """
        if self.forecaster is None:
            if self.model_path.exists():
                self.load_model()
                self._require_training_frame()
                self.intervals_enabled = getattr(self.forecaster, "_cs_df", None) is not None
            else:
                raise ValueError(
                    "Model not trained. Run /pipeline/run before requesting forecasts."
                )

        # At this point forecaster must be loaded
        if self.forecaster is None:
            raise ValueError("Failed to load forecaster model")
        forecaster = self.forecaster  # Local var for type narrowing

        frame = self._require_training_frame()
        h = horizon or self.settings.forecast.horizon
        all_ids = sorted(frame["unique_id"].unique().tolist())
        requested_ids = ids or all_ids
        last_timestamp = pd.Timestamp(frame["ds"].max())

        future_x = self.data_engineer.build_future_exogenous(
            ids=all_ids,
            last_timestamp=last_timestamp,
            horizon=h,
            freq=self.settings.forecast.freq,
        )
        missing = forecaster.get_missing_future(h=h, X_df=future_x)
        if not missing.empty:
            raise ValueError(f"missing future exogenous rows: {len(missing)}")

        _ = forecaster.make_future_dataframe(h=h)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "X does not have valid feature names, but LGBMRegressor was fitted with "
                    "feature names"
                ),
            )
            if self.intervals_enabled:
                predict_levels: list[int | float] | None = list(
                    levels or self.settings.forecast.levels
                )
            else:
                predict_levels = None
            predictions = forecaster.predict(
                h=h,
                X_df=future_x,
                level=predict_levels,
                before_predict_callback=before_predict_cleanup,
                after_predict_callback=after_predict_clip,
            )

        # Reconstruct prices from returns BEFORE ensemble calculation
        # This ensures ensemble is mean of prices, not mean of returns
        if return_prices and self.data_engineer.target_type.value != "price":
            predictions = self._reconstruct_prices(predictions)

        # Add ensemble mean AFTER price reconstruction
        predictions = self._add_ensemble_column(
            predictions, model_names=list(forecaster.models.keys())
        )

        if ids:
            predictions = predictions[predictions["unique_id"].isin(requested_ids)].reset_index(
                drop=True
            )
        return predictions

    def _reconstruct_prices(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """Reconstruct prices from predicted returns."""
        if self._last_prices is None:
            # Try to get from training frame
            frame = self._require_training_frame()
            self._last_prices = self.data_engineer.get_last_prices(frame)

        return self.data_engineer.reconstruct_prices(predictions, self._last_prices)

    @staticmethod
    def _add_ensemble_column(frame: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
        available = [name for name in model_names if name in frame.columns]
        if len(available) < 2:
            return frame
        out = frame.copy()
        out["ensemble_mean"] = out[available].mean(axis=1)
        return out

    def update_with_latest(self, new_observations: pd.DataFrame) -> None:
        if self.forecaster is None:
            raise RuntimeError("forecaster is not fitted")
        self.forecaster.update(new_observations)

    def save_model(self) -> Path:
        if self.forecaster is None:
            raise RuntimeError("forecaster is not fitted")
        model_path = self.model_path
        ensure_directory(model_path.parent)
        self.forecaster.save(model_path)
        return model_path

    @staticmethod
    def _install_numpy_pickle_compat_aliases() -> None:
        """
        Provide module aliases for cross-version NumPy pickle compatibility.

        Some persisted artifacts may reference ``numpy._core`` (NumPy 2) while
        runtime has ``numpy.core`` (NumPy 1.x).
        """
        import sys

        import numpy.core as np_core
        import numpy.core.numeric as np_numeric

        sys.modules.setdefault("numpy._core", np_core)
        sys.modules.setdefault("numpy._core.numeric", np_numeric)

    def load_model(self) -> MLForecast:
        try:
            self.forecaster = MLForecast.load(self.model_path)
        except ModuleNotFoundError as exc:
            if "numpy._core" not in str(exc):
                raise
            self._install_numpy_pickle_compat_aliases()
            self.forecaster = MLForecast.load(self.model_path)
        return self.forecaster

    def get_fitted_values(self) -> pd.DataFrame:
        """Return fitted (in-sample) predictions.

        Tries ``forecast_fitted_values()`` first (available when the model was
        fit in the current session).  Falls back to the persisted parquet file
        produced during training.
        """
        if self.forecaster is not None:
            try:
                return self.forecaster.forecast_fitted_values()
            except Exception:
                pass
        if self.fitted_values_path.exists():
            return load_parquet(self.fitted_values_path)
        raise ValueError(
            "Fitted values not available. The model must be trained with "
            "the current pipeline (run /tasks/full-pipeline) to generate "
            "backtest predictions."
        )

    def run_lightgbm_cv(self, training_frame: pd.DataFrame | None = None) -> pd.DataFrame:
        frame = training_frame if training_frame is not None else self._require_training_frame()
        model_frame = self._model_frame(frame)
        lgb_cv = LightGBMCV(
            freq=self.settings.forecast.freq,
            lags=self.settings.forecast.lags,
            date_features=["dayofweek", "month", "is_month_end"],
            num_threads=self.settings.forecast.num_threads,
        )
        lgb_cv.fit(
            model_frame,
            n_windows=2,
            h=min(5, self.settings.forecast.horizon),
            num_iterations=25,
            static_features=["sector_code", "asset_class_code"],
            metric="mape",
            verbose_eval=False,
            compute_cv_preds=False,
            keep_last_n=self.settings.forecast.keep_last_n,
        )
        cv_forecaster = MLForecast.from_cv(lgb_cv)
        return cv_forecaster.predict(h=min(5, self.settings.forecast.horizon))

    @staticmethod
    def _model_frame(frame: pd.DataFrame) -> pd.DataFrame:
        cols = [
            "unique_id",
            "ds",
            "y",
            "sector_code",
            "asset_class_code",
            "is_weekend",
            "is_month_start",
            "is_month_end",
            "week_of_year",
            "month_sin",
            "month_cos",
        ]
        existing_cols = [col for col in cols if col in frame.columns]
        return frame[existing_cols].copy()

    def run_full_pipeline(self, download: bool = True) -> dict[str, Any]:
        frame = self.prepare_training_data(download=download)
        fitted_values = self.fit(frame)
        cv_df, cv_summary = self.cross_validate(frame)
        forecasts = self.forecast()
        model_path = self.save_model()

        summary = {
            "rows": int(len(frame)),
            "series": int(frame["unique_id"].nunique()),
            "fitted_rows": int(len(fitted_values)),
            "cv_rows": int(len(cv_df)),
            "forecast_rows": int(len(forecasts)),
            "best_model": str(cv_summary.iloc[0]["model"]),
            "model_path": str(model_path),
        }
        save_json(summary, self.run_summary_path)
        save_parquet(cv_summary, self.cv_summary_path)
        return {
            "summary": summary,
            "cv_summary": cv_summary,
            "forecasts": forecasts,
        }

    def get_cv_summary(self, run_if_missing: bool = False, download: bool = False) -> pd.DataFrame:
        if self.latest_cv_summary is not None:
            return self.latest_cv_summary.copy()
        if self.cv_summary_path.exists():
            return load_parquet(self.cv_summary_path)
        if run_if_missing:
            frame = self.prepare_training_data(download=download)
            _, cv_summary = self.cross_validate(frame)
            save_parquet(cv_summary, self.cv_summary_path)
            return cv_summary
        raise RuntimeError("CV summary not found. Run /pipeline/run first.")
