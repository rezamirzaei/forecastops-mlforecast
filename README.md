# MLForecast Real-World Forecasting Project

End-to-end forecasting platform using real-world financial time-series data, `mlforecast`, FastAPI, Angular (MVC-style), Docker, notebooks, and CI/CD.

## Why MLForecast?

**MLForecast** is the ideal choice for this project because it solves key challenges in production time-series forecasting:

| Challenge | MLForecast Solution |
|-----------|-------------------|
| **Multiple models to compare** | Train Ridge, ElasticNet, RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting, LightGBM, XGBoost, and MLP in a single `fit()` call |
| **Feature engineering complexity** | Built-in lag transforms (`RollingMean`, `ExpandingStd`, `ExponentiallyWeightedMean`, `SeasonalRollingMean`) with automatic naming |
| **Cross-validation for time-series** | Native `cross_validation()` with proper temporal splits, refitting, and callbacks |
| **Prediction intervals** | Conformal prediction intervals out-of-the-box with `PredictionIntervals` |
| **Performance at scale** | Multi-threaded feature computation, NumPy backend optimization |
| **Target preprocessing** | `Differences` and `LocalStandardScaler` target transforms handled automatically |
| **Model persistence** | Simple `save()`/`load()` for deployment |

### Compared to Alternatives

- **Prophet**: Single-model, slower on many series, limited ML model support
- **statsforecast**: Great for statistical models, but MLForecast is better for ML ensembles
- **Manual sklearn pipelines**: Require custom lag feature code, CV logic, and multi-model orchestration

## Key Features

### Returns-Based Prediction (New!)

Instead of predicting raw prices directly, this project predicts **log returns** and reconstructs prices:

```
Log Return: r_t = ln(P_t / P_{t-1})
Reconstruction: P_t = P_{t-1} × exp(r_t)
```

**Benefits:**
- More stationary than raw prices (better for ML models)
- Symmetric treatment of gains/losses
- Additive over time periods
- Better for percentage-based error metrics

Configure via `FORECAST__TARGET_TYPE`:
- `log_return` (default, recommended)
- `percent_return`
- `price` (legacy)

### Model Ensemble

9 models trained in parallel with different complexity levels:

| Complexity | Models | Use Case |
|------------|--------|----------|
| **Simple** | Ridge, ElasticNet | Fast, interpretable baseline |
| **Moderate** | RandomForest, ExtraTrees | Feature interactions |
| **Complex** | GB, HGB, LightGBM, XGBoost, MLP | Non-linear patterns |

### Technical Features

Automatically computed per series:
- `volatility_5d`, `volatility_20d`: Rolling return volatility
- `momentum_5d`, `momentum_20d`: Price momentum
- `range_pct`: High-low range as percentage
- `volume_ma_ratio`: Volume vs 20-day moving average

## What This Project Includes

- Real data download from `stooq.com` (multiple tickers).
- Full S&P 500 symbol universe as the default ticker set (`src/mlforecast_realworld/data/sp500.py`).
- Clean, reusable Python package under `src/`.
- Strict Pydantic validation for configs, records, requests, and responses.
- ML pipeline with broad `mlforecast` feature usage.
- API service (FastAPI) + Angular UI (models/services/controllers/views).
- Dockerized backend and frontend with `docker-compose`.
- Unit tests for backend functions/modules.
- Notebooks that call reusable `src` code.
- CI/CD workflow for lint, tests, frontend build, and Docker image build.

## Project structure

```text
.
├── src/mlforecast_realworld/
│   ├── api/
│   ├── data/
│   ├── ml/
│   ├── schemas/
│   ├── utils/
│   ├── cli.py
│   └── config.py
├── tests/
├── notebooks/
├── frontend/angular-ui/
├── docker/
├── docs/
├── docker-compose.yml
├── pyproject.toml
└── .github/workflows/ci.yml
```

## MLForecast features exercised

See `docs/MLFORECAST_FEATURE_COVERAGE.md` for detailed mapping.

Highlights:

- Multi-model forecasting, lags, lag transforms, date features, target transforms.
- Model set includes linear, elastic-net, random-forest, extra-trees, histogram gradient boosting, LightGBM, XGBoost, plus `ensemble_mean`.
- `preprocess`, `fit_models`, `fit`, `predict`, `cross_validation`.
- Fitted values, conformal prediction intervals, callbacks.
- Future frame validation: `make_future_dataframe`, `get_missing_future`.
- Persistence: `save`, `load`.
- Optional LightGBM CV and `MLForecast.from_cv`.

## Quickstart (local)

### 1) Install backend

```bash
python -m pip install -e '.[dev]'
```

### 2) Run checks

```bash
ruff check src tests
pytest
```

### 3) Run pipeline from CLI

```bash
forecast-cli run-all
```

### 4) Start API

```bash
uvicorn mlforecast_realworld.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5) Start Angular UI

```bash
cd frontend/angular-ui
npm ci
npm run build
npm start
```

## API endpoints

- `GET /health`
- `GET /series`
- `POST /pipeline/run?download=true|false`
- `GET /pipeline/metrics?run_if_missing=true|false`
- `POST /forecast`

Call `POST /pipeline/run` before `POST /forecast` to ensure model artifacts are trained and persisted.

By default, `/series` returns the full S&P 500 symbol universe (503 tradable symbols including
multi-class listings like `BRK-B.US` and `GOOGL.US`). You can override via `DATA__TICKERS`.

Example request to `/forecast`:

```json
{
  "horizon": 14,
  "ids": ["AAPL.US", "MSFT.US"],
  "levels": [80, 95]
}
```

## Docker

```bash
docker compose up --build
```

- Backend: `http://localhost:8000`
- Frontend: `http://localhost:4200`

## Notebooks

- `notebooks/01_data_engineering.ipynb`
- `notebooks/02_ml_pipeline_mlforecast.ipynb`

Both notebooks import and use `src/mlforecast_realworld` modules directly.

## CI/CD

GitHub Actions workflow: `.github/workflows/ci.yml`

Stages:

1. Backend lint + tests.
2. Frontend dependency install + production build.
3. Docker image builds for backend and frontend.
