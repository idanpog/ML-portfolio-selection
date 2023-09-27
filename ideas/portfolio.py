import pandas as pd
import numpy as np
from abc import abstractmethod


class Portfolio:
    def __init__(self, weights=None) -> None:
        """A constructor can be called with no parameters.
        Otherwise, it may load a pre-saved weights vector.
        Note: If you use a pre-saved weights, than your submission must include this file.
        """
        super().__init__(weights)
        pass

    @abstractmethod
    def train(self, train_data: pd.DataFrame) -> None:
        """
        Input: train_data: a dataframe as downloaded from yahoo finance,
        containing about 5 years of history, with all the training data.
        The following day (the first that does not appear in the history) is the test day.

        Output (optional): weights vector.
        """
        pass

    @abstractmethod
    def get_portfolio(self, train_data: pd.DataFrame) -> np.array:
        pass

    @abstractmethod
    def get_prices(self, train_data: pd.DataFrame) -> np.array:
        """nearly the same as get_portfolio, just that it returns the expected prices for each stock in each day."""
        pass
