from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

from mlforecast_realworld.config import DataSourceSettings
from mlforecast_realworld.utils.io import ensure_directory, save_parquet

logger = logging.getLogger(__name__)


def ticker_to_yahoo_symbol(ticker: str) -> str:
    """Convert Stooq-style ticker (aapl.us) to Yahoo symbol (AAPL)."""
    # Remove .us suffix and uppercase
    if ticker.lower().endswith(".us"):
        return ticker[:-3].upper()
    return ticker.upper()


def yahoo_symbol_to_ticker(symbol: str) -> str:
    """Convert Yahoo symbol (AAPL) to our standard ticker format (AAPL.US)."""
    return f"{symbol.upper()}.US"


class YahooFinanceDownloader:
    """Download stock data from Yahoo Finance."""

    def __init__(self, settings: DataSourceSettings, output_dir: Path) -> None:
        self.settings = settings
        self.output_dir = output_dir

    def download_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Download data for a single ticker. Returns None if download fails."""
        symbol = ticker_to_yahoo_symbol(ticker)
        try:
            stock = yf.Ticker(symbol)
            start_date = str(self.settings.start_date)
            end_date = str(self.settings.end_date or date.today())

            df = stock.history(start=start_date, end=end_date, auto_adjust=False)

            if df.empty:
                logger.debug("No data for ticker: %s", ticker)
                return None

            # Rename columns to our standard format
            df = df.reset_index()
            df = df.rename(columns={
                "Date": "ds",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            })

            # Select only the columns we need
            df = df[["ds", "open", "high", "low", "close", "volume"]].copy()
            df["unique_id"] = yahoo_symbol_to_ticker(symbol)

            # Ensure ds is datetime without timezone
            df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

            return df[["unique_id", "ds", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.debug("Error downloading %s: %s", ticker, e)
            return None

    def download_all(self, delay_between_requests: float = 0.1) -> pd.DataFrame:
        """Download data for all configured tickers."""
        frames: list[pd.DataFrame] = []
        total = len(self.settings.tickers)
        successful = 0
        failed = 0

        logger.info("Starting download of %d tickers from Yahoo Finance...", total)

        for i, ticker in enumerate(self.settings.tickers):
            if i > 0 and delay_between_requests > 0:
                time.sleep(delay_between_requests)

            frame = self.download_ticker(ticker)
            if frame is not None and len(frame) > 0:
                frames.append(frame)
                successful += 1
            else:
                failed += 1

            # Log progress every 50 tickers
            if (i + 1) % 50 == 0 or i == total - 1:
                logger.info("Progress: %d/%d (successful: %d, failed: %d)",
                            i + 1, total, successful, failed)

        if not frames:
            raise ValueError("No data downloaded for any ticker. Check network connection.")

        logger.info("Download complete: %d successful, %d failed", successful, failed)
        combined = pd.concat(frames, ignore_index=True)
        return combined.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    def save_raw(self, frame: pd.DataFrame, file_name: str = "market_raw.parquet") -> Path:
        ensure_directory(self.output_dir)
        return save_parquet(frame, self.output_dir / file_name)


# Keep StooqDownloader as alias for backward compatibility, but use Yahoo
StooqDownloader = YahooFinanceDownloader


# Legacy functions for tests
def build_download_url(base_url: str, ticker: str, interval: str) -> str:
    """Legacy function for backward compatibility."""
    return f"{base_url}?s={ticker}&i={interval}"


def is_valid_csv_response(csv_text: str) -> bool:
    """Legacy validation function for backward compatibility."""
    if not csv_text or len(csv_text) < 50:
        return False
    if "No data" in csv_text or "Exceeded" in csv_text:
        return False
    if "<html" in csv_text.lower() or "<!doctype" in csv_text.lower():
        return False
    first_line = csv_text.split('\n')[0].strip()
    if "Date" not in first_line or "Close" not in first_line:
        return False
    return True


def parse_stooq_csv(csv_text: str, ticker: str) -> pd.DataFrame:
    """Legacy parse function for backward compatibility."""
    from io import StringIO
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

