"""
Author: Cfir

Market portfolio is a portfolio that holds every asset in the market,
each asset is weighted by its market capitalization.
"""

import pandas as pd
import numpy as np
from ideas.portfolio import Portfolio
import utils


class MarketPortfolio():
    def __init__(self, weights=None) -> None:
        pass

    def train(self, train_data: pd.DataFrame) -> None:
        self.train_data = train_data
        self.last_alocation = self._get_allocation_based_on_market_value(
            self.train_data
        )

        pass

    def get_portfolio(self, train_data: pd.DataFrame) -> np.array:
        self.last_alocation = self._get_allocation_based_on_market_value(
            self.train_data
        )
        return self.last_alocation

    def get_prices(self, train_data: pd.DataFrame) -> np.array:
        # TODO: implement
        print("idk what to do here")
        pass

    def _get_allocation_based_on_market_value(self, df):
        """
        Calculates allocation based on market value.

        Args:
            df: A Pandas DataFrame containing the market values of the assets.

        Returns:
            A Pandas Series containing the allocation of each asset.
        """
        market_values = df["Adj Close"]

        # take last index
        market_values = market_values.iloc[-1]
        # Calculate the total market value.
        total_market_value = market_values.sum()

        # Calculate the allocation of each asset.
        allocation = market_values / total_market_value

        return allocation
