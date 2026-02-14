# Data Engineering Design

## Source

- Real-world market data is downloaded from `stooq.com` for configured tickers.
- Download logic is in `src/mlforecast_realworld/data/downloader.py`.

## Transformations

Implemented in `src/mlforecast_realworld/data/engineering.py`:

- Schema normalization and type coercion.
- Invalid row filtering (missing fields, non-positive price/volume).
- Target creation: `y = close`.
- Static metadata columns (`sector`, `asset_class`) plus numeric encodings.
- Deterministic calendar exogenous features:
  - `is_weekend`
  - `is_month_start`
  - `is_month_end`
  - `week_of_year`
  - `month_sin`
  - `month_cos`
- Data quality report generation.

## Validation

- Row-level validation with Pydantic models in `src/mlforecast_realworld/schemas/records.py`.
- Pipeline writes quality report and run summary JSON artifacts.

## Reusability

- Components are split into downloader, engineer, model factory, and pipeline orchestration.
- All operations are callable from CLI, API, tests, and notebooks.
