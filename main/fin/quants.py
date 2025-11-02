"""
Another helper module to compute quantities relevant to financial portfolio
like weighted average and standard deviation and Sharpe ratio.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from fin.returns import weighted_mean_daily_returns
from fin.finance_data_types import DATA_ARRAY, SERIES_ARRAY, FLOAT, INT_STD, NUMERIC

def weighted_mean_sum(
    means: SERIES_ARRAY[FLOAT], weights: SERIES_ARRAY[FLOAT]
) -> FLOAT:
    
    weighted_mu: FLOAT = float(np.sum(means*weights))
    return weighted_mu

def weighted_std(
    cov_matrix: DATA_ARRAY[FLOAT], weights: SERIES_ARRAY[FLOAT]
) -> FLOAT:
    weighted_sigma: FLOAT = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return weighted_sigma

def sharpe_ratio(
    exp_return: FLOAT, volatility: FLOAT, risk_free_rate: FLOAT = 0.005
) -> FLOAT:
    sharpe_ratio: FLOAT = (exp_return - risk_free_rate) / float(volatility)
    return sharpe_ratio

def value_at_risk(
    investment: NUMERIC, mu: FLOAT, sigma: FLOAT, conf_interval: FLOAT = 0.95
) -> FLOAT:
    if conf_interval >= 1 or conf_interval <= 0:
        raise ValueError("expected confidence level is between 0 and 1!")
    value_at_risk: FLOAT = investment * (mu - sigma * norm.ppf(1 - conf_interval))
    return value_at_risk 
