from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def walk_forward_splits(
    df: pd.DataFrame,
    date_col: str = "date",
    min_train_days: int = 252,
    test_days: int = 21,
):
    dates = np.array(sorted(pd.to_datetime(df[date_col]).unique()))
    start = min_train_days

    while start < len(dates):
        train_end = dates[start - 1]
        test_end = dates[min(start + test_days - 1, len(dates) - 1)]
        train_mask = pd.to_datetime(df[date_col]) <= train_end
        test_mask = (
            (pd.to_datetime(df[date_col]) > train_end) &
            (pd.to_datetime(df[date_col]) <= test_end)
        )
        yield train_mask, test_mask
        start += test_days


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "directional_accuracy": float(np.mean((y_true > 0) == (y_pred > 0))),
        "n": int(len(y_true)),
    }
