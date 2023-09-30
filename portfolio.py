import pandas as pd
import numpy as np


class Portfolio:
    def __init__(self, w=np.nan):
        self.eps = 2.897811214769126
        self.w = 25
        self.train_data_length = None
        self.rel_returns = None
        self.b = None
        self.olmar_rel_returns = None

    def train(self, train_data: pd.DataFrame):
        return None

    def filter_last_k_days(self, df, k):
        return df.tail(k)

    def get_portfolio(self, train_data: pd.DataFrame):
        not_full_train_data = self.filter_last_k_days(train_data, self.w + 1)["Adj Close"]
        tick2df = not_full_train_data.interpolate("linear").bfill().ffill().fillna(1)

        self.olmar_rel_returns = 1 + tick2df.pct_change(1)

        self.rel_returns = self.olmar_rel_returns
        inv_rel_returns = 1 / self.olmar_rel_returns

        olmar_next_price = (1 + np.sum(np.array([np.prod(inv_rel_returns.iloc[-i:], axis=0) for i in range(self.w - 1)]
                                                ),axis=0,)) / self.w

        if self.b is None:
            self.b = np.ones(self.rel_returns.shape[1]) / self.rel_returns.shape[1]
        olmar_b = self.OLMAR(self.eps, self.w, olmar_next_price, self.b)
        self.b = olmar_b
        return self.b

    def OLMAR(self, eps, w, pred_next_price, b_t):
        # calculate average relative predicted price
        avg_pred_price = np.mean(pred_next_price)
        # calculate next Lagrange multiplier
        try:
            next_lam = (eps - b_t @ pred_next_price) / (np.square(
                np.linalg.norm(pred_next_price - avg_pred_price)
            ) + 0.000001)
        except:
            next_lam = 0
        next_lam = max(0.0, next_lam)

        next_b = b_t + next_lam * (pred_next_price - avg_pred_price)
        normalized_b = self.new_simplex_proj(next_b)
        return normalized_b

    def new_simplex_proj(self, y):
        """Projection of y onto simplex."""
        m = len(y)
        bget = False

        s = sorted(y, reverse=True)
        tmpsum = 0.0

        for ii in range(m - 1):
            tmpsum = tmpsum + s[ii]
            tmax = (tmpsum - 1) / (ii + 1)
            if tmax >= s[ii + 1]:
                bget = True
                break

        if not bget:
            tmax = (tmpsum + s[m - 1] - 1) / m

        return np.maximum(y - tmax, 0.0)
