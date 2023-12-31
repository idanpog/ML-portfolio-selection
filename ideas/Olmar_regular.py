import pandas as pd
import numpy as np
from time import time
import matplotlib.pyplot as plt
from prophet import Prophet

import warnings
import logging
from tqdm import tqdm

# logging.getLogger("prophet").setLevel(logging.ERROR)
# logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
# logging.basicConfig(level=logging.ERROR)

warnings.filterwarnings("ignore")
# logging.getLogger('prophet').setLevel(logging.WARNING)
logger = logging.getLogger("cmdstanpy")
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)


class OlmarPortfolio:
    def __init__(self, **kwargs):
        self.eps = 2.897811214769126
        self.w = 25
        self.train_data_length = None
        self.t = 0
        self.rel_returns = None
        self.b = None
        self.is_prophet = False
        self.trained_models = {}
        self.prophet_predictions = None
        self.pred_returns = None
        self.all_prices = None
        self.olmar_rel_returns = None

        # update parameters if provided
        self.__dict__.update(kwargs)

    def train(self, train_data: pd.DataFrame):
        flag = True
        if self.is_prophet:
            data = train_data.reset_index()
            data.rename(columns={"Date/": "ds"}, inplace=True)

            closes = data.Close
            dates = data.Date

            tickers = closes.columns

            data_dict = {
                tick: pd.concat([dates, pd.Series(closes[tick])], axis=1)
                .rename(columns={"Date": "ds", tick: "y"})
                .fillna(0)
                for tick in tickers
            }

            for tick in tqdm(tickers, desc="Training models"):
                self.trained_models[tick] = Prophet().fit(data_dict[tick])
                # self.b = np.ones(len(self.trained_models)) / len(self.trained_models)
                future = self.trained_models[tick].make_future_dataframe(periods=30)
                dates = future["ds"]
                if self.prophet_predictions is None:
                    self.prophet_predictions = pd.DataFrame(dates, columns=["ds"])
                preds = (
                    self.trained_models[tick]
                    .predict(future)[["ds", "yhat"]]
                    .rename(columns={"yhat": f"{tick}"})
                    .set_index("ds")
                )
                if flag:
                    self.prophet_predictions = preds
                    flag = False
                    # self.prophet_predictions[tick] = preds\
                else:
                    self.prophet_predictions = pd.concat(
                        [self.prophet_predictions, preds], axis=1
                    )
                x = 1

            train_prices = train_data.interpolate()["Adj Close"]
            self.train_data_length = train_prices.shape[0]
            self.all_prices = pd.concat(
                [train_prices, self.prophet_predictions], axis=0
            )
            self.rel_returns = 1 + self.all_prices.pct_change(1).dropna()

        return None

    def filter_last_k_days(self, df, k):
        return df.tail(k)

    def get_portfolio(self, train_data: pd.DataFrame):
        # if not self.is_prophet:
        not_full_train_data = self.filter_last_k_days(train_data, self.w+1)["Adj Close"]
        # TODO: check what is the correct axis for interpolating between days and not between stocks
        tick2df = not_full_train_data.interpolate("linear").bfill().ffill().fillna(1)

        self.olmar_rel_returns = 1 + tick2df.pct_change(1)
        if not self.is_prophet:
            self.rel_returns = self.olmar_rel_returns
        inv_rel_returns = 1 / self.olmar_rel_returns

        if self.is_prophet:
            prophet_prices = 1 / self.rel_returns.iloc[self.train_data_length + self.t]
        olmar_next_price = (
            1
            + np.sum(
                np.array(
                    [
                        np.prod(inv_rel_returns.iloc[-i:], axis=0)
                        for i in range(self.w - 1)
                    ]
                ),
                axis=0,
            )
        ) / self.w

        # pred_next_price = (prophet_prices + olmar_next_price) / 2
        if self.b is None:
            self.b = np.ones(self.rel_returns.shape[1]) / self.rel_returns.shape[1]
        olmar_b, olmar_flag = self.OLMAR(self.eps, self.w, olmar_next_price, self.b)
        if self.is_prophet:
            prophet_b, prophet_flag = self.OLMAR(1.2, self.w, prophet_prices, self.b)
        self.b = prophet_b if self.is_prophet else olmar_b
        self.b = olmar_b / 3 + 2 * prophet_b / 3 if self.is_prophet else olmar_b
        self.t += 1
        return self.b

    def OLMAR(self, eps, w, pred_next_price, b_t):
        # calculate average relative predicted price
        # m = pred_next_price.size
        avg_pred_price = np.mean(pred_next_price)

        # calculate next Lagrange multiplier
        try :
            next_lam = (eps - b_t @ pred_next_price) / (np.square(
                np.linalg.norm(pred_next_price - avg_pred_price)
            ) + 0.000001)
        except:
            next_lam =0
        flag = True if next_lam > 0 else False
        next_lam = max(0.0, next_lam)

        next_b = b_t + next_lam * (pred_next_price - avg_pred_price)
        normalized_b = self.new_simplex_proj(next_b)
        return normalized_b, flag

    def illai_simplex_proj(self, x):
        m = x.size

        s = np.sort(x)[::-1]
        c = np.cumsum(s) - 1.0
        h = s - c / (np.arange(m) + 1)

        r = np.max(np.where(h > 0)[0])
        r = np.max(h[0])
        t = c[r] / (r + 1)

        return np.maximum(0, x - t)


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