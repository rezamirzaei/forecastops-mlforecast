#!/usr/bin/env python
"""Run the full pipeline to download data and train models."""
from mlforecast_realworld.config import get_settings
from mlforecast_realworld.ml.pipeline import ForecastPipeline


def main():
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
    pipeline.prepare_training_data(download=True)
    print(f"   Downloaded {len(pipeline.training_frame['unique_id'].unique())} companies")
    print(f"   Total rows: {len(pipeline.training_frame):,}")

    print("2. Training models...")
    pipeline.fit()
    print(f"   Trained models: {list(pipeline.forecaster.models.keys())}")

    print("3. Running cross-validation...")
    pipeline.cross_validate()  # Results saved to artifacts
    print("   CV complete")

    print("4. Saving model...")
    pipeline.save()
    print("   Model saved")

    print()
    print("=== DONE ===")
    print(f"Training data: {pipeline.processed_data_path}")
    print(f"Model: {pipeline.model_path}")

if __name__ == "__main__":
    main()

