import pandas as pd
import yfinance as yf

# from ideas.minvar_portfolio import MinVarPortfolio as Portfolio
# from ideas.prophet_portfolio import ProphetPortfolio as Portfolio
from ideas.Olmar_regular import OlmarPortfolio as Portfolio

# from ideas.olmar import olmar as Portfolio
# from ideas.market_portfolio import MarketPortfolio as Portfolio
import numpy as np
import pickle
import os

# START_DATE = '2017-08-01'
# END_TEST_DATE = '2022-09-30'
# END_TRAIN_DATE = '2022-08-31'

START_DATE = "2016-08-01"
END_TRAIN_DATE = "2023-08-01"
END_TEST_DATE = "2023-08-31"


def get_data(start_date, end_date):

    sanitized_start_date = str(start_date).replace(':', '_').replace(' ', '_')
    sanitized_end_date = str(end_date).replace(':', '_').replace(' ', '_')

    CACHE_FILE = f"s&p500_data_{sanitized_start_date}_{sanitized_end_date}.pkl"
    if os.path.exists(CACHE_FILE):
        # Load data from the cache file if it exists
        with open(CACHE_FILE, "rb") as file:
            data = pickle.load(file)
    else:
        wiki_table = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        )
        sp_tickers = wiki_table[0]
        tickers = [
            ticker.replace(".", "-") for ticker in sp_tickers["Symbol"].to_list()
        ]
        print(f"downloading data for dates {start_date} till {end_date}")
        data = yf.download(tickers, start_date, end_date)

        # Save the data as a Pickle file for future use
        with open(CACHE_FILE, "wb") as file:
            pickle.dump(data, file)

    return data


def test_portfolio(start_date, end_train_date, end_test_date, stg_params=None):
    full_train = get_data(start_date, end_test_date)
    returns = []
    if stg_params is None:
        strategy = Portfolio()
    else:
        strategy = Portfolio(**stg_params)

    # NOTE THAT THIS LINE ISN'T SUPPOSED TO BE HERE
    strategy.train(full_train[full_train.index < end_train_date])
    # NOTE THAT THIS LINE ISN'T SUPPOSED TO BE HERE

    for test_date in pd.date_range(end_train_date, end_test_date):
        if test_date not in full_train.index:
            continue
        train = full_train[full_train.index < test_date]
        cur_portfolio = strategy.get_portfolio(train)
        if not np.isclose(cur_portfolio.sum(), 1):
            raise ValueError(
                f"The sum of the portfolio should be 1, not {cur_portfolio.sum()}"
            )
        test_data = full_train["Adj Close"].loc[test_date].to_numpy()
        prev_test_data = train["Adj Close"].iloc[-1].to_numpy()
        test_data = test_data / prev_test_data - 1
        cur_return = cur_portfolio @ test_data
        returns.append({"date": test_date, "return": cur_return})
    returns = pd.DataFrame(returns).set_index("date")
    mean_return, std_returns = float(returns.mean()), float(returns.std())
    sharpe = mean_return / std_returns
    print("Sharp Ratio: ", sharpe)

    # portfolio variance
    cov_matrix = full_train["Adj Close"].pct_change().cov()
    port_variance = np.dot(cur_portfolio.T, np.dot(cov_matrix, cur_portfolio))
    print("Portfolio Variance: ", port_variance)
    return sharpe, port_variance


if __name__ == "__main__":
    test_portfolio()
