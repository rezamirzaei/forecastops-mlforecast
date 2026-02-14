from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_parquet(df: pd.DataFrame, output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    df.to_parquet(output_path, index=False)
    return output_path


def load_parquet(input_path: Path) -> pd.DataFrame:
    return pd.read_parquet(input_path)


def save_json(payload: dict[str, Any], output_path: Path) -> Path:
    ensure_directory(output_path.parent)
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return output_path
