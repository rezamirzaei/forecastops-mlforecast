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
