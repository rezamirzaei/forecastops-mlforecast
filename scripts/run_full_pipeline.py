#!/usr/bin/env python
"""Run the full pipeline to download data and train models."""

from __future__ import annotations

from mlforecast_realworld.config import get_settings
from mlforecast_realworld.ml.pipeline import ForecastPipeline
from mlforecast_realworld.utils.io import save_json, save_parquet


def main() -> None:
    # Clear cache to reload settings
    get_settings.cache_clear()
    settings = get_settings()

    print("=== Configuration ===")
    print(f"Tickers: {len(settings.data.tickers)} companies")
    print(f"Start date: {settings.data.start_date}")
    print()

    print("=== Running Pipeline ===")
    pipeline = ForecastPipeline(settings)

    print("1. Downloading data...")
    frame = pipeline.prepare_training_data(download=True)
    print(f"   Downloaded {frame['unique_id'].nunique()} companies")
    print(f"   Total rows: {len(frame):,}")

    print("2. Training models...")
    fitted = pipeline.fit(frame)
    model_names = list(pipeline.forecaster.models.keys()) if pipeline.forecaster else []
    print(f"   Trained models: {model_names}")
    print(f"   Fitted rows: {len(fitted):,}")

    print("3. Running cross-validation...")
    cv_df, cv_summary = pipeline.cross_validate(frame)
    save_parquet(cv_summary, pipeline.cv_summary_path)
    print(f"   CV rows: {len(cv_df):,}")
    if not cv_summary.empty:
        best = cv_summary.iloc[0]
        print(f"   Best model: {best['model']} (sMAPE={best['smape']:.4f})")

    print("4. Generating forecasts...")
    forecasts = pipeline.forecast()
    print(f"   Forecast rows: {len(forecasts):,}")

    print("5. Saving model...")
    model_path = pipeline.save_model()
    print(f"   Model saved to {model_path}")

    summary = {
        "rows": int(len(frame)),
        "series": int(frame["unique_id"].nunique()),
        "fitted_rows": int(len(fitted)),
        "cv_rows": int(len(cv_df)),
        "forecast_rows": int(len(forecasts)),
        "best_model": str(cv_summary.iloc[0]["model"]) if not cv_summary.empty else "unknown",
        "model_path": str(model_path),
    }
    save_json(summary, pipeline.run_summary_path)

    print()
    print("=== DONE ===")
    print(f"Training data: {pipeline.processed_data_path}")
    print(f"Model: {pipeline.model_path}")
    print(f"CV summary: {pipeline.cv_summary_path}")
    print(f"Run summary: {pipeline.run_summary_path}")

if __name__ == "__main__":
    main()
