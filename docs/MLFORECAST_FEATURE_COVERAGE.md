# MLForecast Feature Coverage

## Why MLForecast for This Project?

We chose **MLForecast** over alternatives (Prophet, statsforecast, manual sklearn) for these key reasons:

1. **Multi-Model Training**: Train 7+ models (linear, tree, boosting) in one `fit()` call
2. **Automatic Feature Engineering**: Built-in lag transforms eliminate manual feature code
3. **Production-Ready**: Native model persistence, prediction intervals, and CV support
4. **Performance**: Multi-threaded feature computation, NumPy backend optimization
5. **Flexibility**: Custom date features, callbacks, and sklearn-compatible models

## Features Used in This Project

This project exercises the following `mlforecast` APIs in `src/mlforecast_realworld/ml/pipeline.py` and `src/mlforecast_realworld/ml/factory.py`:

### MLForecast Construction

- `MLForecast(...)` with:
  - Multiple models: `LinearRegression`, `ElasticNet`, `RandomForestRegressor`, `ExtraTreesRegressor`, `HistGradientBoostingRegressor`, `LGBMRegressor`, `XGBRegressor`
  - `lags`: [1, 7, 14, 21] for capturing short and medium-term patterns
  - `lag_transforms`: `RollingMean`, `RollingStd`, `ExpandingMean`, `ExpandingStd`, `ExponentiallyWeightedMean`, `SeasonalRollingMean`
  - `date_features`: dayofweek, month, quarter, custom `week_of_month` callable
  - `target_transforms`: `Differences([1])`, `LocalStandardScaler()`
  - Custom `lag_transforms_namer` for consistent feature naming

### Training & Inference

- `preprocess(...)` for feature inspection and `return_X_y=True` mode
- `fit_models(...)` for manual training path
- `fit(...)` with conformal `PredictionIntervals`
- `forecast_fitted_values()` for in-sample predictions
- `predict(...)` with `X_df`, `ids`, and `level` parameters

### Cross-Validation

- `cross_validation(...)` with callbacks and interval levels
- `cross_validation_fitted_values()` for CV fitted values
- `before_predict_callback` and `after_predict_callback` for data cleaning

### Future Prediction

- `make_future_dataframe(...)` for generating future dates
- `get_missing_future(...)` for validating exogenous completeness
- `update(...)` via `update_with_latest(...)` for online updates

### Persistence & Advanced

- `save(...)` and `load(...)` for model deployment
- `LightGBMCV` + `MLForecast.from_cv(...)` for optimized LightGBM training

## Verification

- Unit tests cover forecasting orchestration in `tests/test_pipeline.py`
- Forecast prechecks ensure future exogenous completeness before inference
- CI enforces lint, tests, and container image builds
