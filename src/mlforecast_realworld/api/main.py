from __future__ import annotations

from fastapi import FastAPI, HTTPException

from mlforecast_realworld.api.service import ForecastService
from mlforecast_realworld.schemas.records import ForecastRequest, PipelineSummary


def create_app() -> FastAPI:
    app = FastAPI(
        title="MLForecast Real-World API",
        version="0.1.0",
        description="Forecasting API powered by mlforecast and real market data.",
    )
    service = ForecastService()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/pipeline/run", response_model=PipelineSummary)
    def run_pipeline(download: bool = True) -> PipelineSummary:
        return service.run_pipeline(download=download)

    @app.get("/pipeline/metrics")
    def pipeline_metrics(run_if_missing: bool = True, download: bool = False) -> dict[str, object]:
        metrics = service.get_metrics(run_if_missing=run_if_missing, download=download)
        best = metrics[0]["model"] if metrics else None
        return {"metrics": metrics, "best_model": best, "count": len(metrics)}

    @app.post("/forecast")
    def forecast(request: ForecastRequest) -> dict[str, object]:
        try:
            records = service.get_forecast(request)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"records": records, "count": len(records)}

    return app


app = create_app()
