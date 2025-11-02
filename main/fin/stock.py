import numpy as np
import pandas as pd

from sklearn.metrics import r2_score

from .asset import Asset
from .finance_data_types import FLOAT, SERIES_DATA, STD_ARRAY

class Stock(Asset):
    investmentinfo: SERIES_DATA
    beta: FLOAT | None
    cov_matrix: STD_ARRAY | None
    rsquared: FLOAT | None

    def __init__(self, investmentinfo: SERIES_DATA, data: SERIES_DATA) -> None:

        self.name = investmentinfo.Name
        self.investmentinfo = investmentinfo
        super().__init__(data, self.name, asset_type="Stock")
        self.beta = None
        self.rsquared = None

    def comp_cov_matrix(self, market_daily_returns: SERIES_DATA) -> STD_ARRAY:
        cov_mat = np.cov(
            self.comp_daily_returns(), 
            market_daily_returns.to_frame()[market_daily_returns.name],
        )
        self.cov_matrix = cov_mat
        return cov_mat
    
    def comp_beta(self, market_daily_returns: SERIES_DATA) -> FLOAT:
        cm = self.comp_cov_matrix(market_daily_returns)
        beta = float(cm[0, 1] / cm[1, 1])
        self.beta = beta
        return beta
    
    def comp_rsquared(self, market_daily_returns: SERIES_DATA) -> FLOAT:
        rsquared = float(
            r2_score(
                market_daily_returns.to_frame()[market_daily_returns.name]
            )
        )
        self.rsquared = rsquared
        return rsquared
