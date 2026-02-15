#!/usr/bin/env python
"""Download all S&P 500 data and run the ML pipeline."""
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
    """Download data and run pipeline."""
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
    logger.info("Starting data download and pipeline...")
    logger.info("This may take several minutes for 500+ companies...")

    try:
        pipeline.run_full_pipeline(download=True)
        logger.info("Pipeline completed successfully!")

        # Show results
        if pipeline.training_frame is not None:
            companies = pipeline.training_frame["unique_id"].nunique()
            rows = len(pipeline.training_frame)
            logger.info(f"Training data: {rows:,} rows, {companies} companies")
            logger.info(f"Companies: {sorted(pipeline.training_frame['unique_id'].unique().tolist())[:20]}...")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()

