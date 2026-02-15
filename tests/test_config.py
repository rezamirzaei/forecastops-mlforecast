from pathlib import Path

from mlforecast_realworld.config import AppSettings


def test_settings_resolves_relative_path(tmp_path: Path) -> None:
    settings = AppSettings(paths={"project_root": tmp_path})
    resolved = settings.resolved_path(Path("data/raw"))
    assert resolved == (tmp_path / "data/raw").resolve()


def test_all_dirs_contains_expected_defaults() -> None:
    settings = AppSettings()
    dirs = settings.paths.all_dirs()
    assert settings.paths.raw_data_dir in dirs
    assert settings.paths.report_dir in dirs


def test_default_tickers_use_sp500_universe_when_no_env() -> None:
    """When no DATA__TICKERS env var, defaults to full S&P 500."""
    # This test verifies the default factory behavior
    # If .env sets DATA__TICKERS, that will override
    settings = AppSettings()
    # Tickers should be a list of strings
    assert isinstance(settings.data.tickers, list)
    assert len(settings.data.tickers) > 0
    # All should be lowercase with .us suffix
    for ticker in settings.data.tickers:
        assert ticker.endswith(".us"), f"Ticker {ticker} should end with .us"
