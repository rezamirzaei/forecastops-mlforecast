# MLForecast Feature Coverage

This project exercises the following `mlforecast` APIs and capabilities in `src/mlforecast_realworld/ml/pipeline.py` and `src/mlforecast_realworld/ml/factory.py`:

- `MLForecast(...)` construction with:
  - multiple models (`LinearRegression`, `RandomForestRegressor`, `LGBMRegressor`, `XGBRegressor`)
  - `lags`
  - `lag_transforms`
  - `date_features` (including custom callable)
  - `target_transforms`
  - custom `lag_transforms_namer`
- `preprocess(...)` both for inspection and `return_X_y=True`
- `fit_models(...)` manual training path
- `fit(...)` with conformal `PredictionIntervals`
- `forecast_fitted_values()`
- `cross_validation(...)` with callbacks and interval levels
- `cross_validation_fitted_values()`
- `make_future_dataframe(...)`
- `get_missing_future(...)`
- `predict(...)` using `X_df`, `ids`, and `level`
- `update(...)` via `update_with_latest(...)`
- `save(...)` and `load(...)`
- `LightGBMCV` + `MLForecast.from_cv(...)`

Verification points:

- Unit tests cover forecasting orchestration in `tests/test_pipeline.py`.
- Forecast prechecks ensure future exogenous completeness before inference.
- CI enforces lint, tests, and container image builds.
