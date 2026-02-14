from __future__ import annotations

import json

import typer

from mlforecast_realworld.ml.pipeline import ForecastPipeline

app = typer.Typer(help="Real-world mlforecast pipeline CLI")


def _get_pipeline() -> ForecastPipeline:
    return ForecastPipeline()


@app.command("download")
def download_data() -> None:
    pipeline = _get_pipeline()
    frame = pipeline.prepare_training_data(download=True)
    typer.echo(
        f"Prepared training data with {len(frame)} rows and "
        f"{frame['unique_id'].nunique()} series"
    )


@app.command("train")
def train_model(download: bool = False) -> None:
    pipeline = _get_pipeline()
    frame = pipeline.prepare_training_data(download=download)
    fitted = pipeline.fit(frame)
    pipeline.save_model()
    typer.echo(f"Model trained. Fitted rows: {len(fitted)}")


@app.command("forecast")
def make_forecast(horizon: int = 14) -> None:
    pipeline = _get_pipeline()
    if pipeline.processed_data_path.exists() and pipeline.model_path.exists():
        pipeline.load_model()
        pipeline.training_frame = pipeline._require_training_frame()
    else:
        pipeline.run_full_pipeline(download=True)
    preds = pipeline.forecast(horizon=horizon)
    typer.echo(preds.head().to_string(index=False))


@app.command("run-all")
def run_all(download: bool = True) -> None:
    pipeline = _get_pipeline()
    result = pipeline.run_full_pipeline(download=download)
    payload = result["summary"]
    typer.echo(json.dumps(payload, indent=2))


@app.command("lgbm-cv")
def lgbm_cv(download: bool = False) -> None:
    pipeline = _get_pipeline()
    frame = pipeline.prepare_training_data(download=download)
    preds = pipeline.run_lightgbm_cv(frame)
    typer.echo(preds.head().to_string(index=False))


if __name__ == "__main__":
    app()
