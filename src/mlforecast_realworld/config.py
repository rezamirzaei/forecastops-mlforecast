"""
Application configuration using Pydantic Settings.

This module provides typed configuration classes that can be loaded
from environment variables and .env files.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSourceSettings(BaseModel):
    """Configuration for data sources."""

    tickers: list[str] = Field(
        default_factory=lambda: ["aapl.us", "msft.us", "goog.us", "amzn.us", "meta.us"]
    )
    start_date: date = date(2019, 1, 1)
    end_date: date | None = None
    interval: Literal["d", "w", "m"] = "d"
    base_url: HttpUrl = "https://stooq.com/q/d/l/"


class PathsSettings(BaseModel):
    """Configuration for file paths."""

    project_root: Path = Path(".")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    model_dir: Path = Path("artifacts/models")
    report_dir: Path = Path("artifacts/reports")

    def all_dirs(self) -> list[Path]:
        """Return all configured directories."""
        return [self.raw_data_dir, self.processed_data_dir, self.model_dir, self.report_dir]


class ForecastSettings(BaseModel):
    """Configuration for forecasting parameters."""

    freq: str = "B"
    horizon: PositiveInt = 14
    season_length: PositiveInt = 5
    lags: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 3, 5, 7, 14, 21])
    cv_windows: PositiveInt = 2
    cv_step_size: PositiveInt = 7
    levels: list[int] = Field(default_factory=lambda: [80, 95])
    keep_last_n: PositiveInt = 450
    enable_prediction_intervals: bool = False
    num_threads: PositiveInt = 1
    random_state: int = 42
    target_type: Literal["price", "log_return", "percent_return"] = "log_return"


class APISettings(BaseModel):
    """Configuration for API server."""

    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:4200", "http://localhost:8080"]
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


class AppSettings(BaseSettings):
    """Main application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    data: DataSourceSettings = DataSourceSettings()
    paths: PathsSettings = PathsSettings()
    forecast: ForecastSettings = ForecastSettings()
    api: APISettings = APISettings()

    def resolved_path(self, path: Path) -> Path:
        """Resolve a path relative to project root."""
        if path.is_absolute():
            return path
        return (self.paths.project_root / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Get cached application settings."""
    return AppSettings()
