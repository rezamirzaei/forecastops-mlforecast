from mlforecast_realworld.ml.pipeline import ForecastPipeline

if __name__ == "__main__":
    pipeline = ForecastPipeline()
    result = pipeline.run_full_pipeline(download=True)
    print(result["summary"])
