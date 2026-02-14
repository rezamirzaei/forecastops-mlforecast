from pathlib import Path

import pandas as pd

from mlforecast_realworld.utils.io import ensure_directory, load_parquet, save_json, save_parquet


def test_ensure_directory_creates_path(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "dir"
    created = ensure_directory(target)
    assert created.exists()
    assert created.is_dir()


def test_save_and_load_parquet_roundtrip(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]})
    file_path = tmp_path / "frame.parquet"
    save_parquet(frame, file_path)
    loaded = load_parquet(file_path)
    pd.testing.assert_frame_equal(frame, loaded)


def test_save_json_writes_payload(tmp_path: Path) -> None:
    path = tmp_path / "payload.json"
    save_json({"ok": True}, path)
    assert path.read_text(encoding="utf-8").strip().startswith("{")
