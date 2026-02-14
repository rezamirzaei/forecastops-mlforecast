from datetime import datetime

import pandas as pd
import pytest

from mlforecast_realworld.schemas.records import (
    ForecastRecord,
    MarketRecord,
    forecast_records_from_frame,
    validate_market_rows,
)


def test_market_record_validation() -> None:
    record = MarketRecord(
        unique_id="AAPL.US",
        ds=datetime(2024, 1, 2),
        open=10,
        high=12,
        low=9,
        close=11,
        volume=100,
        y=11,
        sample_weight=0.5,
        sector="Technology",
        asset_class="equity",
        sector_code=1,
        asset_class_code=1,
        is_weekend=0,
        is_month_start=0,
        is_month_end=0,
        week_of_year=1,
    )
    assert record.unique_id == "AAPL.US"


def test_validate_market_rows_empty_raises() -> None:
    with pytest.raises(ValueError):
        validate_market_rows(pd.DataFrame())


def test_forecast_records_from_frame() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["AAPL.US"],
            "ds": [pd.Timestamp("2024-01-10")],
            "lin_reg": [123.4],
            "rf": [124.0],
        }
    )
    records = forecast_records_from_frame(frame, ["lin_reg", "rf"])
    assert len(records) == 2
    assert isinstance(records[0], ForecastRecord)
