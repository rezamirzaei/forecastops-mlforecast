#!/usr/bin/env python
"""
Run the ML forecasting pipeline with progress reporting.

This script runs the full pipeline (data prep, training, evaluation)
and saves all artifacts for future use. If artifacts already exist,
it will skip the pipeline and use existing artifacts.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --download  # To download fresh data
    python scripts/run_pipeline.py --force     # Force re-run even if artifacts exist
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

# Setup logging with progress info
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def print_progress(step: int, total: int, message: str) -> None:
    """Print a progress indicator."""
    bar_length = 30
    filled = int(bar_length * step / total)
    bar = "█" * filled + "░" * (bar_length - filled)
    percent = 100 * step / total
    print(f"\r[{bar}] {percent:5.1f}% | {message}", end="", flush=True)
    if step == total:
        print()  # New line when complete


def check_existing_artifacts(pipeline: Any) -> dict[str, bool]:
    """Check which artifacts already exist."""
    return {
        "model": pipeline.model_path.exists(),
        "data": pipeline.processed_data_path.exists(),
        "cv_summary": pipeline.cv_summary_path.exists(),
        "report": pipeline.report_path.exists(),
    }


def load_existing_artifacts(pipeline: Any) -> dict[str, Any]:
    """Load existing artifacts and return summary."""
    import pandas as pd

    print("\n" + "=" * 60)
    print("  Loading Existing Artifacts")
    print("=" * 60 + "\n")

    start_time = time.time()

    # Load training data
    logger.info("Loading training data from %s", pipeline.processed_data_path)
    frame = pd.read_parquet(pipeline.processed_data_path)
    pipeline.training_frame = frame
    logger.info("  ✓ Loaded %d rows, %d companies", len(frame), frame["unique_id"].nunique())

    # Load model
    logger.info("Loading model from %s", pipeline.model_path)
    pipeline.load_model()
    model_names = list(pipeline.forecaster.models.keys()) if pipeline.forecaster else []
    logger.info("  ✓ Loaded models: %s", ", ".join(model_names))

    # Load CV summary if exists
    best_model = "unknown"
    best_smape = 0.0
    if pipeline.cv_summary_path.exists():
        cv_summary = pd.read_parquet(pipeline.cv_summary_path)
        if len(cv_summary) > 0:
            best_model = cv_summary.iloc[0]["model"]
            best_smape = cv_summary.iloc[0]["smape"]
            logger.info("  ✓ CV summary loaded: best=%s (sMAPE=%.4f)", best_model, best_smape)

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 60)
    print("  Artifacts Loaded Successfully!")
    print("=" * 60)
    print(f"\n  Summary:")
    print(f"    • Data rows:      {len(frame):,}")
    print(f"    • Companies:      {frame['unique_id'].nunique()}")
    print(f"    • Date range:     {frame['ds'].min().date()} to {frame['ds'].max().date()}")
    print(f"    • Models:         {len(model_names)}")
    print(f"    • Best model:     {best_model} (sMAPE: {best_smape:.4f})")
    print(f"    • Load time:      {total_time:.1f}s")
    print(f"\n  Artifacts loaded from:")
    print(f"    • Model:    {pipeline.model_path}")
    print(f"    • Data:     {pipeline.processed_data_path}")
    print(f"    • CV:       {pipeline.cv_summary_path}")
    print()

    return {
        "summary": {
            "rows": len(frame),
            "companies": frame["unique_id"].nunique(),
            "models": model_names,
            "best_model": best_model,
            "best_smape": best_smape,
            "load_time_seconds": total_time,
            "loaded_from_cache": True,
        },
        "paths": {
            "model": str(pipeline.model_path),
            "data": str(pipeline.processed_data_path),
            "cv_summary": str(pipeline.cv_summary_path),
        },
    }


def run_pipeline(download: bool = False, force: bool = False) -> dict[str, Any]:
    """Run the full pipeline with progress reporting."""
    from mlforecast_realworld.config import get_settings
    from mlforecast_realworld.ml.pipeline import ForecastPipeline

    # Get settings
    settings = get_settings()
    logger.info(f"Configuration: {len(settings.data.tickers)} tickers configured")

    # Initialize pipeline
    pipeline = ForecastPipeline(settings)

    # Check for existing artifacts
    existing = check_existing_artifacts(pipeline)
    has_all_artifacts = existing["model"] and existing["data"]

    if has_all_artifacts and not force and not download:
        logger.info("Found existing artifacts - loading from cache")
        logger.info("  (Use --force to re-run pipeline, or --download for fresh data)")
        return load_existing_artifacts(pipeline)

    if has_all_artifacts and not force:
        logger.info("Found existing artifacts but --download specified - will update data")

    print("\n" + "=" * 60)
    print("  MLForecast Pipeline Runner")
    print("=" * 60 + "\n")

    total_steps = 5
    start_time = time.time()

    # Step 1: Prepare training data
    print_progress(1, total_steps, "Preparing training data...")
    logger.info("Step 1/5: Preparing training data (download=%s)", download)
    step_start = time.time()

    frame = pipeline.prepare_training_data(download=download)

    logger.info(
        "  ✓ Training data ready: %d rows, %d companies (%.1fs)",
        len(frame),
        frame["unique_id"].nunique(),
        time.time() - step_start,
    )

    # Step 2: Fit models
    print_progress(2, total_steps, "Training models...")
    logger.info("Step 2/5: Training models")
    step_start = time.time()

    fitted_values = pipeline.fit(frame)

    model_names = list(pipeline.forecaster.models.keys()) if pipeline.forecaster else []
    logger.info(
        "  ✓ Models trained: %s (%.1fs)",
        ", ".join(model_names),
        time.time() - step_start,
    )

    # Step 3: Cross-validation
    print_progress(3, total_steps, "Running cross-validation...")
    logger.info("Step 3/5: Running cross-validation")
    step_start = time.time()

    cv_df, cv_summary = pipeline.cross_validate(frame)

    best_model = cv_summary.iloc[0]["model"]
    best_smape = cv_summary.iloc[0]["smape"]
    logger.info(
        "  ✓ Cross-validation complete: best=%s (sMAPE=%.4f) (%.1fs)",
        best_model,
        best_smape,
        time.time() - step_start,
    )

    # Step 4: Generate forecasts
    print_progress(4, total_steps, "Generating forecasts...")
    logger.info("Step 4/5: Generating forecasts")
    step_start = time.time()

    forecasts = pipeline.forecast()

    logger.info(
        "  ✓ Forecasts generated: %d predictions (%.1fs)",
        len(forecasts),
        time.time() - step_start,
    )

    # Step 5: Save artifacts
    print_progress(5, total_steps, "Saving artifacts...")
    logger.info("Step 5/5: Saving artifacts")
    step_start = time.time()

    model_path = pipeline.save_model()

    logger.info("  ✓ Model saved to: %s (%.1fs)", model_path, time.time() - step_start)

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print(f"\n  Summary:")
    print(f"    • Data rows:      {len(frame):,}")
    print(f"    • Companies:      {frame['unique_id'].nunique()}")
    print(f"    • Date range:     {frame['ds'].min().date()} to {frame['ds'].max().date()}")
    print(f"    • Models trained: {len(model_names)}")
    print(f"    • Best model:     {best_model} (sMAPE: {best_smape:.4f})")
    print(f"    • Forecast rows:  {len(forecasts):,}")
    print(f"    • Total time:     {total_time:.1f}s")
    print(f"\n  Artifacts saved to:")
    print(f"    • Model:    {model_path}")
    print(f"    • Data:     {pipeline.processed_data_path}")
    print(f"    • Report:   {pipeline.report_path}")
    print(f"    • CV:       {pipeline.cv_summary_path}")
    print()

    return {
        "summary": {
            "rows": len(frame),
            "companies": frame["unique_id"].nunique(),
            "models": model_names,
            "best_model": best_model,
            "best_smape": best_smape,
            "forecast_rows": len(forecasts),
            "total_time_seconds": total_time,
            "loaded_from_cache": False,
        },
        "paths": {
            "model": str(model_path),
            "data": str(pipeline.processed_data_path),
            "report": str(pipeline.report_path),
            "cv_summary": str(pipeline.cv_summary_path),
        },
    }


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run the MLForecast pipeline with progress reporting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py              # Use cached artifacts if available
  python scripts/run_pipeline.py --download   # Download fresh data and retrain
  python scripts/run_pipeline.py --force      # Force re-run even with cached artifacts
  python scripts/run_pipeline.py -d -f        # Download fresh data and force retrain
        """,
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download fresh data from Yahoo Finance",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-run pipeline even if artifacts exist",
    )
    args = parser.parse_args()

    try:
        run_pipeline(download=args.download, force=args.force)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

