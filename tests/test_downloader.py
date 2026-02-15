from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from mlforecast_realworld.config import DataSourceSettings
from mlforecast_realworld.data.downloader import (
    StooqDownloader,
    YahooFinanceDownloader,
    build_download_url,
    is_valid_csv_response,
    parse_stooq_csv,
    ticker_to_yahoo_symbol,
    yahoo_symbol_to_ticker,
)

SAMPLE_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,100,101,99,100.5,12345
2024-01-03,101,102,100,101.5,23456
"""


def test_ticker_to_yahoo_symbol() -> None:
    assert ticker_to_yahoo_symbol("aapl.us") == "AAPL"
    assert ticker_to_yahoo_symbol("MSFT.US") == "MSFT"
    assert ticker_to_yahoo_symbol("GOOGL") == "GOOGL"


def test_yahoo_symbol_to_ticker() -> None:
    assert yahoo_symbol_to_ticker("AAPL") == "AAPL.US"
    assert yahoo_symbol_to_ticker("msft") == "MSFT.US"


def test_build_download_url() -> None:
    url = build_download_url("https://stooq.com/q/d/l/", "aapl.us", "d")
    assert url == "https://stooq.com/q/d/l/?s=aapl.us&i=d"


def test_is_valid_csv_response_with_valid_data() -> None:
    assert is_valid_csv_response(SAMPLE_CSV) is True


def test_is_valid_csv_response_with_no_data() -> None:
    assert is_valid_csv_response("No data") is False


def test_is_valid_csv_response_with_exceeded() -> None:
    assert is_valid_csv_response("Exceeded the daily hits limit") is False


def test_is_valid_csv_response_with_html() -> None:
    assert is_valid_csv_response("<html><body>Error</body></html>") is False


def test_is_valid_csv_response_with_empty() -> None:
    assert is_valid_csv_response("") is False
    assert is_valid_csv_response(None) is False


def test_parse_stooq_csv() -> None:
    frame = parse_stooq_csv(SAMPLE_CSV, "aapl.us")
    assert list(frame.columns) == ["unique_id", "ds", "open", "high", "low", "close", "volume"]
    assert frame["unique_id"].iloc[0] == "AAPL.US"


def test_parse_stooq_csv_missing_columns_raises() -> None:
    with pytest.raises(ValueError):
        parse_stooq_csv("Date,Open\n2024-01-01,10", "aapl.us")


def test_stooq_downloader_is_yahoo_downloader() -> None:
    """Verify StooqDownloader is aliased to YahooFinanceDownloader."""
    assert StooqDownloader is YahooFinanceDownloader


def test_yahoo_downloader_download_ticker(tmp_path: Path) -> None:
    """Test downloading a single ticker with mocked yfinance."""
    settings = DataSourceSettings(
        tickers=["aapl.us"],
        start_date="2024-01-01",
        end_date="2024-01-10",
    )
    downloader = YahooFinanceDownloader(settings=settings, output_dir=tmp_path)

    # Create mock data
    mock_df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "Open": [100.0, 101.0],
        "High": [101.0, 102.0],
        "Low": [99.0, 100.0],
        "Close": [100.5, 101.5],
        "Volume": [12345, 23456],
    }).set_index("Date")

    with patch("mlforecast_realworld.data.downloader.yf.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        result = downloader.download_ticker("aapl.us")

        assert result is not None
        assert len(result) == 2
        assert result["unique_id"].iloc[0] == "AAPL.US"
        assert list(result.columns) == ["unique_id", "ds", "open", "high", "low", "close", "volume"]


def test_yahoo_downloader_download_all(tmp_path: Path) -> None:
    """Test downloading multiple tickers."""
    settings = DataSourceSettings(
        tickers=["aapl.us", "msft.us"],
        start_date="2024-01-01",
        end_date="2024-01-10",
    )
    downloader = YahooFinanceDownloader(settings=settings, output_dir=tmp_path)

    mock_df = pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
        "Open": [100.0, 101.0],
        "High": [101.0, 102.0],
        "Low": [99.0, 100.0],
        "Close": [100.5, 101.5],
        "Volume": [12345, 23456],
    }).set_index("Date")

    with patch("mlforecast_realworld.data.downloader.yf.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = mock_df
        mock_ticker_class.return_value = mock_ticker

        result = downloader.download_all(delay_between_requests=0)

        assert result["unique_id"].nunique() == 2
        assert len(result) == 4


def test_yahoo_downloader_handles_empty_response(tmp_path: Path) -> None:
    """Test that empty responses are handled gracefully."""
    settings = DataSourceSettings(
        tickers=["invalid.us"],
        start_date="2024-01-01",
        end_date="2024-01-10",
    )
    downloader = YahooFinanceDownloader(settings=settings, output_dir=tmp_path)

    with patch("mlforecast_realworld.data.downloader.yf.Ticker") as mock_ticker_class:
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty response
        mock_ticker_class.return_value = mock_ticker

        result = downloader.download_ticker("invalid.us")
        assert result is None


def test_save_raw(tmp_path: Path) -> None:
    settings = DataSourceSettings(tickers=["aapl.us"])
    downloader = YahooFinanceDownloader(settings=settings, output_dir=tmp_path)
    frame = parse_stooq_csv(SAMPLE_CSV, "aapl.us")
    path = downloader.save_raw(frame)
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 2
