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
    INT,
    DICT_LIST_PASS,
    NUMERIC,


)