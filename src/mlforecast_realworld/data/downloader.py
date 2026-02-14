from __future__ import annotations

from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from mlforecast_realworld.config import DataSourceSettings
from mlforecast_realworld.utils.io import ensure_directory, save_parquet


def build_download_url(base_url: str, ticker: str, interval: str) -> str:
    return f"{base_url}?s={ticker}&i={interval}"


def parse_stooq_csv(csv_text: str, ticker: str) -> pd.DataFrame:
    frame = pd.read_csv(StringIO(csv_text))
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns in stooq payload: {sorted(missing)}")
    frame = frame.rename(
        columns={
            "Date": "ds",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    frame["unique_id"] = ticker.upper()
    frame["ds"] = pd.to_datetime(frame["ds"], utc=False)
    return frame[["unique_id", "ds", "open", "high", "low", "close", "volume"]]


class StooqDownloader:
    def __init__(self, settings: DataSourceSettings, output_dir: Path) -> None:
        self.settings = settings
        self.output_dir = output_dir
        self.session = requests.Session()

    def download_ticker(self, ticker: str) -> pd.DataFrame:
        url = build_download_url(str(self.settings.base_url), ticker, self.settings.interval)
        response = self.session.get(url, timeout=60)
        response.raise_for_status()
        ticker_frame = parse_stooq_csv(response.text, ticker)
        start = pd.Timestamp(self.settings.start_date)
        end = pd.Timestamp(self.settings.end_date or date.today())
        mask = (ticker_frame["ds"] >= start) & (ticker_frame["ds"] <= end)
        return ticker_frame.loc[mask].reset_index(drop=True)

    def download_all(self) -> pd.DataFrame:
        frames = [self.download_ticker(ticker) for ticker in self.settings.tickers]
        if not frames:
            raise ValueError("no tickers configured")
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def save_raw(self, frame: pd.DataFrame, file_name: str = "market_raw.parquet") -> Path:
        ensure_directory(self.output_dir)
        return save_parquet(frame, self.output_dir / file_name)
