"""
- This file contains the code for various useful tools related to portfolio optimisation
such as constructing the portfolio calculating the asset mean returns and the efficient
frontier for the provided assets.

- As this project focuses mainly on the quantum algorithm to optimize such a portfolio,
the class 'Portfolio' will follow the standard way to calculate the daily/cumlative sum
of returns, Expected returns, value at risk, Volatility, sharpe ratio. And to solve the
optimisaton problem of a minimisation problem - we use quantum algorithms.

- A classical approach will be added for comparison of the quantum solution and completeness
will be added soon.

- An example to each use case can be found in the ./notebooks

- a public class Portfolio
- a public function build_portfolio()
- if the data is not provided by the user of type `pandas.Dataframe`, it is retrieved 
through `yfinance`.

"""

import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

from .fin.efficient_frontier import EfficientFrontier
from .fin.finance_data_types import (
    STD_ARRAY,
    ELEMENT_TYPE, 
    FLOAT,
    INT_STD,
    DICT_LIST_PASS,
    NUMERIC,
    STD_DATETIME,
)

from .fin.quants import (
    weighted_mean_sum,
    weighted_std,
    weighted_mean_daily_returns,
    sharpe_ratio,
    value_at_risk
)

from .fin.returns import (
    daily_returns,
    daily_log_returns,
    historical_mean_return,
    weighted_mean_daily_returns
)

from .fin.asset import Asset
from .fin.stock import Stock


class Portfolio:
    portfolio: pd.DataFrame
    stocks: Dict[str, Stock]
    data: pd.DataFrame
    expected_return: FLOAT
    volatility: FLOAT
    var: FLOAT
    sharpe_ratio: FLOAT
    __total_investment: NUMERIC
    __var_confidence_level: FLOAT
    __risk_free_rate: FLOAT
    __freq: INT_STD
    beta_stocks: pd.DataFrame
    beta: FLOAT | None
    rsquared_stocks: pd.DataFrame
    rsquared: FLOAT | None

    def __init__(self) -> None:

        self.portfolio = pd.DataFrame()
        self.stocks = {}
        self.data = pd.DataFrame()
        self.__var_confidence_level = 0.95
        self.__risk_free_rate = 0.005
        self.__freq = 252
        self.beta_stocks = pd.DataFrame(index=['beta'])
        self.beta = None
        self.rsquared_stocks = pd.DataFrame(index=['rsquared'])
        self.rsquared = None

    
    @property
    def totalinvestment(self) -> NUMERIC:
        return self.__total_investment
    
    @totalinvestment.setter
    def totalinvestment(self, val: NUMERIC) -> None:
        if val is not None:
            if not isinstance(val, (FLOAT, INT_STD)):
                raise ValueError("Total investment must be a float or integer.")
            if val < 0:
                raise ValueError("The money to be invested should be countable and hence > 0")
            self.__total_investment = val
    
    @property
    def freq(self) -> INT_STD:
        return self.__freq
    
    @freq.setter
    def freq(self, val: INT_STD)

    