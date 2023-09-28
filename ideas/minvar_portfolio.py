import pandas as pd
import numpy as np
from ideas.portfolio import Portfolio
import utils


class MinVarPortfolio(Portfolio):
    def __init__(self, weights=np.nan) -> None:
        """
        A constructor can be called with no parameters.
        Otherwise, it may load a pre-saved weights vector.
        Note: If you use a pre-saved weights, then your submission must include this file.
        """
        self.min_var_portfolio = None


    def train_prev(self, tick2df: dict) -> None:
        """
        Input: train_data: a dataframe as downloaded from yahoo finance,
        containing about 5 years of history, with all the training data.
        The following day (the first that does not appear in the history) is the test day.

        Output (optional): weights vector.
        """
        # calculate the relative return for each stock
        prices = tick2df['Adj Close'].interpolate('linear').dropna(axis=1)#.fillna(method='ffill').fillna(method='bfill')
        returns = prices.pct_change(1)
        c = returns.cov()
        e = np.ones(len(c))
        x_min_weights = (np.linalg.inv(c) @ e) / (e.T @ np.linalg.inv(c) @ e)
        self.min_var_portfolio = x_min_weights

    def train(self, tick2df: dict) -> None:
        # Process prices and drop stocks with NaN values
        prices = tick2df['Adj Close'].interpolate('linear').dropna(axis=1)
        returns = prices.pct_change(1)

        # Compute covariance and portfolio weights for valid stocks
        c = returns.cov()
        e = np.ones(len(c))
        x_min_weights_valid = (np.linalg.inv(c) @ e) / (e.T @ np.linalg.inv(c) @ e)

        # Initialize full weights vector with zeros
        x_min_weights = np.zeros(len(tick2df['Adj Close'].columns))

        # Assign computed weights to valid stocks
        valid_positions = tick2df['Adj Close'].columns.get_indexer(prices.columns)
        x_min_weights[valid_positions] = x_min_weights_valid

        # Assuming the function is inside a class
        self.min_var_portfolio = x_min_weights


    def get_portfolio(self, train_data: pd.DataFrame) -> np.array:
        """
        The function generates model's portfolio for the next day.
        Input: train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history, with all the training data. The following day (the first that does not appear in the history) is the test day.
        Output: numpy vector of stocks allocations.
        Note: Entries must match original order in the input dataframe!
        """
        return self.min_var_portfolio
