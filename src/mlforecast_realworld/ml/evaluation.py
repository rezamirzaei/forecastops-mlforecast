from __future__ import annotations

import numpy as np
import pandas as pd


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    denominator = np.where(denominator == 0, 1, denominator)
    return float(np.mean(np.abs(y_true - y_pred) / denominator) * 100)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return 0.0
    return float(np.abs(y_true - y_pred).sum() / denominator * 100)


def summarize_cv(cv_df: pd.DataFrame) -> pd.DataFrame:
    model_cols = [
        col
        for col in cv_df.columns
        if col not in {"unique_id", "ds", "cutoff", "y"}
        and "-lo-" not in col
        and "-hi-" not in col
    ]
    rows: list[dict[str, float | str]] = []
    y_true = cv_df["y"].to_numpy()
    for model in model_cols:
        y_pred = cv_df[model].to_numpy()
        rows.append(
            {
                "model": model,
                "smape": smape(y_true, y_pred),
                "wape": wape(y_true, y_pred),
            }
        )
    return pd.DataFrame(rows).sort_values("smape").reset_index(drop=True)
