"""
Author: Cfir

Market portfolio is a portfolio that holds every asset in the market,
each asset is weighted by its market capitalization.
"""

import pandas as pd
import numpy as np
from ideas.portfolio import Portfolio
import utils


class MarketPortfolio(Portfolio):
    def __init__(self, weights=None) -> None:
        super().__init__(weights)

    def train(self, tick2df: dict) -> None:
        # TODO: implement

        pass

    def get_portfolio(self, train_data: pd.DataFrame) -> np.array:
        # TODO: implement
        pass

    def get_prices(self, train_data: pd.DataFrame) -> np.array:
        # TODO: implement
        print("idk what to do here")
        pass
