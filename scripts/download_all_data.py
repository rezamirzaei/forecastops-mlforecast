#!/usr/bin/env python
"""
Download all S&P 500 data and pre-train the ML model.

This script initializes the system with all data and a trained model,
so the API can serve predictions immediately without timeout issues.

Usage:
    python scripts/download_all_data.py

The script will:
1. Download historical data for all S&P 500 companies from Yahoo Finance
2. Process and engineer features
3. Train the forecasting models
4. Save all artifacts (data, model, metrics)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def main():
    """Download data and train model."""
    # Import here to ensure logging is set up first
    from mlforecast_realworld.config import get_settings
    from mlforecast_realworld.ml.pipeline import ForecastPipeline

    # Clear cached settings to get fresh config
    get_settings.cache_clear()
    settings = get_settings()

    logger.info(f"Configuration loaded: {len(settings.data.tickers)} tickers")
    logger.info(f"Date range: {settings.data.start_date} to {settings.data.end_date or 'today'}")

    # Create pipeline
    pipeline = ForecastPipeline(settings)

    # Run full pipeline with download
    logger.info("=" * 60)
    logger.info("STEP 1: Downloading data from Yahoo Finance")
    logger.info("=" * 60)
    logger.info("This may take several minutes for 500+ companies...")

    try:
        # Step 1: Download and prepare data
        frame = pipeline.prepare_training_data(download=True)
        companies = frame["unique_id"].nunique()
        rows = len(frame)
        logger.info(f"✓ Downloaded data: {rows:,} rows, {companies} companies")

        # Step 2: Train models
        logger.info("=" * 60)
        logger.info("STEP 2: Training forecasting models")
        logger.info("=" * 60)

        fitted_values = pipeline.fit(frame)
        logger.info(f"✓ Models trained, fitted values: {len(fitted_values):,} rows")

        # Step 3: Cross-validation
        logger.info("=" * 60)
        logger.info("STEP 3: Running cross-validation")
        logger.info("=" * 60)

        cv_df, cv_summary = pipeline.cross_validate(frame)
        best_model = cv_summary.iloc[0]["model"]
        best_smape = cv_summary.iloc[0]["smape"]
        logger.info(f"✓ Cross-validation complete")
        logger.info(f"  Best model: {best_model} (sMAPE: {best_smape:.4f})")

        # Step 4: Save model
        logger.info("=" * 60)
        logger.info("STEP 4: Saving artifacts")
        logger.info("=" * 60)

        model_path = pipeline.save_model()
        logger.info(f"✓ Model saved to: {model_path}")

        # Summary
        logger.info("=" * 60)
        logger.info("INITIALIZATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Companies: {companies}")
        logger.info(f"  Data points: {rows:,}")
        logger.info(f"  Date range: {frame['ds'].min()} to {frame['ds'].max()}")
        logger.info(f"  Best model: {best_model}")
        logger.info("")
        logger.info("The API is now ready to serve predictions!")
        logger.info("Start the API with: uvicorn mlforecast_realworld.api.main:app")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

