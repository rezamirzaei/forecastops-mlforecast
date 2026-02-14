import numpy as np
import pandas as pd

from mlforecast_realworld.ml.evaluation import smape, summarize_cv, wape


def test_smape() -> None:
    y_true = np.array([100, 200])
    y_pred = np.array([110, 190])
    value = smape(y_true, y_pred)
    assert value > 0


def test_wape_zero_denominator() -> None:
    y_true = np.array([0.0, 0.0])
    y_pred = np.array([1.0, 2.0])
    assert wape(y_true, y_pred) == 0.0


def test_summarize_cv() -> None:
    cv_df = pd.DataFrame(
        {
            "unique_id": ["A", "A"],
            "ds": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "cutoff": pd.to_datetime(["2023-12-31", "2023-12-31"]),
            "y": [10.0, 12.0],
            "lin_reg": [10.5, 11.5],
            "rf": [9.0, 14.0],
        }
    )
    summary = summarize_cv(cv_df)
    assert list(summary.columns) == ["model", "smape", "wape"]
    assert summary.iloc[0]["model"] in {"lin_reg", "rf"}
