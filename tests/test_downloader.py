from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from mlforecast_realworld.config import DataSourceSettings
from mlforecast_realworld.data.downloader import (
    StooqDownloader,
    build_download_url,
    is_valid_csv_response,
    parse_stooq_csv,
)

SAMPLE_CSV = """Date,Open,High,Low,Close,Volume
2024-01-02,100,101,99,100.5,12345
2024-01-03,101,102,100,101.5,23456
"""


@dataclass
class DummyResponse:
    text: str

    def raise_for_status(self) -> None:
        return None


def test_build_download_url() -> None:
    url = build_download_url("https://stooq.com/q/d/l/", "aapl.us", "d")
    assert url == "https://stooq.com/q/d/l/?s=aapl.us&i=d"


def test_is_valid_csv_response_with_valid_data() -> None:
    assert is_valid_csv_response(SAMPLE_CSV) is True


def test_is_valid_csv_response_with_no_data() -> None:
    assert is_valid_csv_response("No data") is False


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


def test_download_all_uses_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    settings = DataSourceSettings(
        tickers=["aapl.us", "msft.us"],
        start_date="2024-01-01",
        end_date="2024-01-10",
    )
    downloader = StooqDownloader(settings=settings, output_dir=tmp_path)

    def fake_get(url: str, timeout: int) -> DummyResponse:  # noqa: ARG001
        assert "?s=" in url
        return DummyResponse(text=SAMPLE_CSV)

    monkeypatch.setattr(downloader.session, "get", fake_get)
    frame = downloader.download_all(delay_between_requests=0)  # No delay for testing
    assert frame["unique_id"].nunique() == 2
    assert len(frame) == 4


def test_download_ticker_handles_invalid_response(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Test that invalid responses are handled gracefully."""
    settings = DataSourceSettings(
        tickers=["invalid.us"],
        start_date="2024-01-01",
        end_date="2024-01-10",
    )
    downloader = StooqDownloader(settings=settings, output_dir=tmp_path)

    def fake_get(url: str, timeout: int) -> DummyResponse:  # noqa: ARG001
        return DummyResponse(text="No data")

    monkeypatch.setattr(downloader.session, "get", fake_get)
    result = downloader.download_ticker("invalid.us")
    assert result is None


def test_save_raw(tmp_path: Path) -> None:
    settings = DataSourceSettings(tickers=["aapl.us"])
    downloader = StooqDownloader(settings=settings, output_dir=tmp_path)
    frame = parse_stooq_csv(SAMPLE_CSV, "aapl.us")
    path = downloader.save_raw(frame)
    assert path.exists()
    loaded = pd.read_parquet(path)
    assert len(loaded) == 2
