#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/data/raw /app/data/processed /app/artifacts/models /app/artifacts/reports

# Check if artifacts already exist
MODEL_EXISTS=false
DATA_EXISTS=false

if [ -d "/app/artifacts/models/mlforecast_model" ]; then
    MODEL_EXISTS=true
    echo "✓ Found existing model at /app/artifacts/models/mlforecast_model"
fi

if [ -f "/app/data/processed/market_training.parquet" ]; then
    DATA_EXISTS=true
    echo "✓ Found existing training data at /app/data/processed/market_training.parquet"
fi

# Decide whether to initialize
if [ "$MODEL_EXISTS" = true ] && [ "$DATA_EXISTS" = true ]; then
    echo "✓ Using existing artifacts - skipping initialization"
    echo "  (Set FORCE_INIT=true to force re-initialization)"

    if [ "$FORCE_INIT" = "true" ] || [ "$FORCE_INIT" = "1" ]; then
        echo "FORCE_INIT is set - running pipeline anyway..."
        python /app/scripts/run_pipeline.py --force
    fi
elif [ "$INIT_DATA" = "true" ] || [ "$INIT_DATA" = "1" ]; then
    echo "Initializing data and model..."
    python /app/scripts/run_pipeline.py --download
else
    echo "No existing artifacts found."
    echo "  - Set INIT_DATA=true to download data and train model on startup"
    echo "  - Or use the UI to trigger pipeline manually"
fi

# Execute the main command
exec "$@"


