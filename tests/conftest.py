from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlforecast_realworld.data.engineering import MarketDataEngineer


@pytest.fixture()
def sample_raw_frame() -> pd.DataFrame:
    dates = pd.date_range("2023-01-02", periods=180, freq="B")
    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(123)
    for unique_id, base in [("AAPL.US", 150.0), ("MSFT.US", 250.0), ("GOOG.US", 100.0)]:
        noise = rng.normal(loc=0.0, scale=1.0, size=len(dates)).cumsum()
        close = base + noise
        for ds, value in zip(dates, close, strict=True):
            rows.append(
                {
                    "unique_id": unique_id,
                    "ds": ds,
                    "open": float(value + 0.5),
                    "high": float(value + 1.2),
                    "low": float(value - 1.1),
                    "close": float(value),
                    "volume": int(abs(value) * 1_000),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture()
def sample_training_frame(sample_raw_frame: pd.DataFrame) -> pd.DataFrame:
    engineer = MarketDataEngineer()
    return engineer.build_training_frame(sample_raw_frame)
