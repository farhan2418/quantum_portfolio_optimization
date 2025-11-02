"A simple module to compute different types of returns used in 'quants.py'"
from typing import Any, Union

import numpy as np
import pandas as pd

from .finance_data_types import FLOAT, INT_STD, SERIES_ARRAY, SERIES_DATA

def daily_returns(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.pct_change()
        .dropna(how='all')
        .replace([np.inf, -np.inf], np.nan)
        .astype(np.float64)
    )

def weighted_mean_daily_returns(data: pd.DataFrame, weights: SERIES_ARRAY[FLOAT]
                                ) -> np.ndarray[FLOAT, Any]:
    mean: np.ndarray[FLOAT, Any] = np.dot(daily_returns(data), weights)
    return mean

def daily_log_returns(data: pd.DataFrame) -> pd.DataFrame:
    return np.log(1 + daily_returns(data)).dropna(how='all').astype(np.float64)

def historical_mean_return(data: SERIES_DATA, freq: INT_STD = 252) -> pd.Series:
    return daily_returns(data).mean() * freq