"""
- This module contains the code for various useful tools related to portfolio optimisation
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
    SERIES_DATA
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
    # other statistical moments such as skewness and kurtosis will be added later
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
    def freq(self, val: INT_STD) -> None:
        if not isinstance(val, INT_STD) and val <= 0:
            raise ValueError("Time frequency must be of type integer \
                              and value > 0.")
        self.__freq = val
        self._update()
    
    @property
    def risk_free_rate(self) -> FLOAT:
        return self.__risk_free_rate
    
    @risk_free_rate.setter
    def risk_free_rate(self, val: FLOAT) -> None:
        if not isinstance(val, FLOAT):
            raise ValueError("Risk free rate must be of type float")
        self.__risk_free_rate = val
        self._update()

    @property
    def var_confidence_level(self) -> FLOAT:
        return self.__var_confidence_level
    
    @var_confidence_level.setter
    def var_confidence_level(self, val: FLOAT) -> None:
        if (not isinstance(val, FLOAT)) and (val >= 1 or val <= 0):
            raise ValueError("confidence level must be a type of float \
                             and a value between 0 and 1")
        self.__var_confidence_level = val
        self._update()

    
    def add_stock(self, stock: Stock, defer_update: bool = False) -> None:
        # detailed description of this method

        self.stocks.update({stock.name: stock})
        
        self.portfolio = pd.concat(
            [self.portfolio, stock.investmentinfo.to_frame().T], ignore_index=True
        )

        self.portfolio.name = "Allocation of stocks"
        self._add_stock_data(stock)

        if not defer_update:
            self._update()
    
    def _add_stock_data(self, stock: Stock) -> None:
        self.data.insert(
            loc=len(self.data.columns), column= stock.name, value=stock.data
        )
        self.data.set_index(self.data.index.values, inplace=True)

        self.data.index.rename("Date", inplace=True)

        # there is a confusion whether to add market index and the beta information for the stock
        # as these details are not required for the simple/initial minimisation problem 

    def _update(self) -> None:
        if not (self.portfolio.emoty or not self.stocks or self.data.empty):
            self.totalinvestment = self.portfolio.Allocation.sum()
            self.expected_return = self.comp_expected_return(freq = self.freq)
            self.volatility = self.comp_volatility(freq = self.freq)
            self.var = self.comp_var()
            self.sharpe_ratio = self.comp_sharpe()

    def get_stock(self, name: str) -> Stock:
        return self.stocks[name]
    
    # def comp_cumulative_returns(self) -> pd.DataFrame:
    #     return cum

    def comp_daily_returns(self) -> pd.DataFrame:
        return daily_returns(self.data)
    
    def comp_daily_log_returns(self) -> pd.DataFrame:
        return daily_returns(self.data)
    
    def comp_mean_returns(self, freq: INT_STD = 252) -> pd.Series:
        return historical_mean_return(self.data, freq=freq)
    
    def comp_stock_volatility(self, freq: INT_STD = 252) -> pd.Series:
        return self.comp_daily_returns().std() * np.sqrt(252)

    def comp_weights(self) -> pd.Series:
        return (self.portfolio["Allocation"] / self.totalinvestment).astype(np.float64)
    
    def comp_expected_return(self, freq: INT_STD = 252) -> FLOAT:
        pf_return_means: pd.Series = self.comp_mean_returns()
        weights: pd.Series = self.comp_weights()
        expected_return: FLOAT = weighted_mean_sum(pf_return_means.values, weights)
        self.expected_return = expected_return
        return expected_return

    def comp_volatility(self, freq: INT_STD = 252) -> FLOAT:
        volatility: FLOAT = weighted_std(
            self.comp_cov(), self.comp_weights()
        ) * np.sqrt(freq)
        self.volatility = volatility
        return volatility

    def comp_cov(self) -> pd.DataFrame:
        returns = self.comp_daily_returns()
        return returns.cov()
    
    def comp_var(self) -> FLOAT:
        var: FLOAT = value_at_risk(
            investment= self.totalinvestment,
            mu = self.expected_return,
            sigma = self.volatility,
            conf_interval=self.var_confidence_level
        )
        self.var = var
        return var
    ##  Eficient Frontier is pending

_all_in = lambda l_1, l_2: all(ele in l_2 for ele in l_1)
_complement = lambda set_a, set_b: STD_ARRAY(set(set_b) - set(set_a))

def _yfinance_request(
        names: STD_ARRAY, 
        start_date: STD_DATETIME, 
        end_date: STD_DATETIME
    ) -> SERIES_DATA:
    try:
        import yfinance
    except ImportError:
        print(
            "Could'nt perform Api request to yfinance, \nkindly install the package `yfinance\n"
            + "hint: try the following command `pip install yfinance`"
        )
    
    try:
        if isinstance(start_date, str):
            start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError:
            raise ValueError(
            "Please provide valid values for <start_date> and <end_date> "
            "(either as datetime object or as String in the format '%Y-%m-%d')."
        ) 
    try:
        res: SERIES_DATA = yfinance.download(names, start=start_date, end=end_date)
        if not isinstance(res.columns. pd.multiIndex) and len(names) > 0:
            stock_tuples = [(col, names[0]) for col in list(res.columns)]
            res.columns = pd.MultiIndex.from_tuples(stock_tuples)

    except Exception as ex:
        err: str = (
            "Error occured while retrieving from y yfinance\n"
            + str(ex)
        )
        raise ValueError(err) from ex

def _generate_pfa(names: STD_ARRAY[str] | None = None, 
                  data: SERIES_DATA | None = None
    ) -> SERIES_DATA:
    stock_names: List[str]

    if data is not None:
        err: str = (
            "'data' pandas.DataFrame contains conflicting column labels."
            + "\nMultiple columns with a substring of\n {}\n"
            + "were found. You have two options:"
            + "\n 1. call 'build_portfolio' and pass a pandas.DataFrame "
            + "'pf_allocation' that contains the weights/allocation of stocks "
            + "within your portfolio. 'build_portfolio' will then extract the "
            + "columns from 'data' that match the values of the column 'Name' in "
            + "the pandas.DataFrame 'pf_allocation'."
            + "\n 2. call 'build_portfolio' and pass a pandas.DataFrame 'data' "
            + "that does not have conflicting column labels, e.g. 'GOOG' and "
            + "'GOOG - Adj. Close' are considered conflicting column headers."
        )
        stock_names = data.columns.tolist()

        redundant_stocks: STD_ARRAY = [name.split("-")[0].strip() for name in stock_names]

        seen_names = set()
        for name in redundant_stocks:
            if name in seen_names:
                err = err.format(str(name))
                raise ValueError(err)
            seen_names.add(name)
        
    elif names is not None:
        stock_names = names
    stocks_len = len(stock_names)
    weights = [1.0 / float(stocks_len) for _ in range(stocks_len)]
    return pd.DataFrame({"Allocation": weights, "Name": stock_names})


def _build_portfolio_from_df(data: SERIES_DATA, pf_allocation: SERIES_DATA | None = None) -> Portfolio:
    data = data.astype(np.float64)
    if pf_allocation is None:
        pf_allocation = _generate_pfa(data = data)


    pf: Portfolio = Portfolio()

    for id in range(len(pf_allocation)):
        name: str = pf_allocation.iloc[id].Name
        stock_data: SERIES_DATA = data.loc[:, [name]].copy(deep=True).squeeze()

        pf.add_stock(
            Stock(investmentinfo =pf_allocation.iloc[id], data=stock_data), defer_update= True
        )

    pf._update()
    return pf

def _build_portfolio_from_api(names: STD_ARRAY, pf_allocation: pd.DataFrame | None  = None, \
                              start_date: STD_DATETIME = None, end_date: STD_DATETIME = None, \
                                data_api: str = 'yfinance'
) -> Portfolio:
    # right now this only suppoorts yahoo finance.
    # other apis to be integrated later.
    stock_data: pd.DataFrame
    if data_api == 'yfinance':
        stock_data = _yfinance_request(list(names), start_date, end_date)
    else:
        raise ValueError(
            f"Value of {data_api} is not supported as of now"  + "choose `yfinance`."
        )

    pf: Portfolio = _build_portfolio_from_df(
        stock_data, pf_allocation
    )
    
def build_portfolio(**kwargs: Dict[str, Any]) -> Portfolio:
    required_args = str(
    )
    input_args: List[str] =[
        "pf_allocation", "names", "start_date", "end_date", "data_api"
    ]

    input_error = str(
        "{} is an unsupported keyword. \n{}\nThese are the only supported keywords."
    )
    if not kwargs:
        raise ValueError(
            "\nbuild_portfolio() is missing the required arguments"
        )

   

    if not _all_in(kwargs.keys(), input_args):
        unsupported_input: List[str] = _complement(input_args, kwargs.keys())
        raise ValueError(
            input_error.format(unsupported_input, input_args)
        )

    pf: Portfolio = Portfolio()

    names = cast(List[str], list(kwargs.get("names", [] )))
    pf_allocation = kwargs.get("pf_allocation", None)
    start_date = cast(Optional[STD_DATETIME], kwargs.get("start_date", None))
    end_date = cast(Optional[STD_DATETIME], kwargs.get("end_date", None))
    data_api = cast(str, kwargs.get("data_api", "quandl"))

    pf = _build_portfolio_from_api(
        names=names,
        pf_allocation=pf_allocation,
        start_date=start_date,
        end_date=end_date,
        data_api=data_api,
    )

    return pf

