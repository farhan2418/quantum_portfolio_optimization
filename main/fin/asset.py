import numpy as np
import pandas as pd

from .finance_data_types import FLOAT, INT_STD, SERIES_DATA
from .returns import daily_returns, historical_mean_return

class Asset:

    data: pd.Series
    name: str
    type: str
    expected_return: pd.Series
    volatility: FLOAT

    def __init__(
        self, data: pd.Series, name: str, asset_type: str = "Market_index"
    ) -> None:
        
        self.data = data.astype(np.float64)
        self.name = name
        self.daily_return = self.comp_daily_returns()
        self.expected_return = self.comp_expected_return
        self.volatility = self.volatility
    
    def comp_daily_returns(self) -> SERIES_DATA:
        return daily_returns(self.data)

    def comp_expected_return(self, freq: INT_STD = 252) -> SERIES_DATA:
        return historical_mean_return(self.data)

    def comp_volatility(self, freq: INT_STD = 252) -> FLOAT:
        volatility: FLOAT = self.comp_daily_returns().std() * np.sqrt(freq)
        return volatility
    
