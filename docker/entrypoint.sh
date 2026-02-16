#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/data/raw /app/data/processed /app/artifacts/models /app/artifacts/reports

# Check if we should initialize data on startup
if [ "$INIT_DATA" = "true" ] || [ "$INIT_DATA" = "1" ]; then
    echo "Initializing data and model..."
    # Check if model already exists (from a mounted volume)
    if [ -d "/app/artifacts/models/mlforecast_model" ] && [ -f "/app/data/processed/market_training.parquet" ]; then
        echo "Model and data already exist, skipping initialization."
    else
        echo "Running data download and model training..."
        python /app/scripts/download_all_data.py
    fi
fi

# Execute the main command
exec "$@"

