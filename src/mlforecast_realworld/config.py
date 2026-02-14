from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl, PositiveInt
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataSourceSettings(BaseModel):
    tickers: list[str] = Field(
        default_factory=lambda: ["aapl.us", "msft.us", "goog.us", "amzn.us", "meta.us"]
    )
    start_date: date = date(2015, 1, 1)
    end_date: date | None = None
    interval: Literal["d", "w", "m"] = "d"
    base_url: HttpUrl = "https://stooq.com/q/d/l/"


class PathsSettings(BaseModel):
    project_root: Path = Path(".")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    model_dir: Path = Path("artifacts/models")
    report_dir: Path = Path("artifacts/reports")

    def all_dirs(self) -> list[Path]:
        return [self.raw_data_dir, self.processed_data_dir, self.model_dir, self.report_dir]


class ForecastSettings(BaseModel):
    freq: str = "B"
    horizon: PositiveInt = 14
    lags: list[PositiveInt] = Field(default_factory=lambda: [1, 2, 3, 5, 7, 14, 21])
    cv_windows: PositiveInt = 3
    cv_step_size: PositiveInt = 7
    levels: list[int] = Field(default_factory=lambda: [80, 95])
    keep_last_n: PositiveInt = 650
    num_threads: PositiveInt = 1
    random_state: int = 42


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    environment: Literal["dev", "test", "prod"] = "dev"
    data: DataSourceSettings = DataSourceSettings()
    paths: PathsSettings = PathsSettings()
    forecast: ForecastSettings = ForecastSettings()

    def resolved_path(self, path: Path) -> Path:
        if path.is_absolute():
            return path
        return (self.paths.project_root / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()
