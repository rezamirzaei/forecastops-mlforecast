from __future__ import annotations

import logging
import time
from datetime import date
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from mlforecast_realworld.config import DataSourceSettings
from mlforecast_realworld.utils.io import ensure_directory, save_parquet

logger = logging.getLogger(__name__)


def build_download_url(base_url: str, ticker: str, interval: str) -> str:
    return f"{base_url}?s={ticker}&i={interval}"


def parse_stooq_csv(csv_text: str, ticker: str) -> pd.DataFrame:
    """Parse Stooq CSV response into a DataFrame."""
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

    def download_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Download data for a single ticker. Returns None if download fails."""
        url = build_download_url(str(self.settings.base_url), ticker, self.settings.interval)
        try:
            response = self.session.get(url, timeout=60)
            response.raise_for_status()

            # Check if response contains actual data (not an error page)
            if "No data" in response.text or len(response.text) < 100:
                logger.warning("No data available for ticker: %s", ticker)
                return None

            ticker_frame = parse_stooq_csv(response.text, ticker)

            # Filter by date range
            start = pd.Timestamp(self.settings.start_date)
            end = pd.Timestamp(self.settings.end_date or date.today())
            mask = (ticker_frame["ds"] >= start) & (ticker_frame["ds"] <= end)
            filtered = ticker_frame.loc[mask].reset_index(drop=True)

            if len(filtered) == 0:
                logger.warning("No data in date range for ticker: %s", ticker)
                return None

            return filtered

        except requests.RequestException as e:
            logger.warning("Network error downloading %s: %s", ticker, e)
            return None
        except ValueError as e:
            logger.warning("Parse error for %s: %s", ticker, e)
            return None

    def download_all(self, delay_between_requests: float = 0.1) -> pd.DataFrame:
        """Download data for all configured tickers with rate limiting."""
        frames: list[pd.DataFrame] = []
        total = len(self.settings.tickers)
        successful = 0
        failed = 0

        logger.info("Starting download of %d tickers...", total)

        for i, ticker in enumerate(self.settings.tickers):
            if i > 0 and delay_between_requests > 0:
                time.sleep(delay_between_requests)  # Rate limiting

            frame = self.download_ticker(ticker)
            if frame is not None:
                frames.append(frame)
                successful += 1
            else:
                failed += 1

            # Log progress every 50 tickers
            if (i + 1) % 50 == 0 or i == total - 1:
                logger.info("Progress: %d/%d (successful: %d, failed: %d)",
                            i + 1, total, successful, failed)

        if not frames:
            raise ValueError("No data downloaded for any ticker")

        logger.info("Download complete: %d successful, %d failed", successful, failed)
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def save_raw(self, frame: pd.DataFrame, file_name: str = "market_raw.parquet") -> Path:
        ensure_directory(self.output_dir)
        return save_parquet(frame, self.output_dir / file_name)
