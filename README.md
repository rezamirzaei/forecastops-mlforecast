# MLForecast Real-World Forecasting Project

End-to-end forecasting platform using real-world financial time-series data, `mlforecast`, FastAPI, Angular (MVC-style), Docker, notebooks, and CI/CD.

## What this project includes

- Real data download from `stooq.com` (multiple tickers).
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
- `POST /pipeline/run?download=true|false`
- `POST /forecast`

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

