import pandas as pd
import numpy as np
from prophet import Prophet
from ideas.portfolio import Portfolio
import utils
import ujson as json
from prophet.serialize import model_to_dict, model_from_dict

class ProphetPortfolio(Portfolio):
    def __init__(self, weights=None) -> None:
        """
        A constructor can be called with no parameters.
        Otherwise, it may load a pre-saved weights vector.
        Note: If you use a pre-saved weights, then your submission must include this file.
        """
        if weights:
            self.trained_models = {sym : model_from_dict(w) for sym, w in weights.items()}




    def train(self, tick2df: dict) ->:
        """
        Input: train_data: a dataframe as downloaded from yahoo finance,
        containing about 5 years of history, with all the training data.
        The following day (the first that does not appear in the history) is the test day.

        Output (optional): weights vector.
        """
        self.trained_models = {sym: Prophet().fit(df) for sym, df in tick2df.items()}
        self.b = np.ones(len(self.trained_models)) / len(self.trained_models)

        # save_models
        weights = {sym : model_to_dict(model) for sym, model in self.trained_models.items()}
        with open("saved_models", "w") as f:
            f.write(json.dumps(weights))
        return weights


    def predict(self, sym, date):
        """
        Input: sym: a string representing the ticker of the stock.
        date: a string representing the date to predict.
        Output: a float representing the predicted price of the stock.
        """
        return self.trained_models[sym].predict(date)


    def get_portfolio(self, train_data: pd.DataFrame) -> np.array:
        """
        The function generates model's portfolio for the next day.
        Input: train_data: a dataframe as downloaded from yahoo finance, containing about 5 years of history, with all the training data. The following day (the first that does not appear in the history) is the test day.
        Output: numpy vector of stocks allocations.
        Note: Entries must match original order in the input dataframe!
        """
        train_dates = train_data.index.to_list()
        test_date = len(train_dates) + 1
        x = self.get_prices(train_data, test_date)

        lam = 1/3
        b1 = b + lam*(x-np.mean(x))
        b1 = b1/np.sum(b1)
        self.b = b1
        return b1


        # TODO
        # return weights
        # option 1 : zero-one on the best stock
        # option 2 : [predict for stock / sum of predictions]

    def get_prices(self, train_data: pd.DataFrame, test_date) -> np.array:
        x = [(self.predict(sym, test_date), sym) for sym in self.trained_models.keys()]
        x = np.array(x)



