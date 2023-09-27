import numpy as np
import pandas as pd
from ideas.prophet_portfolio import ProphetPortfolio as Portfolio
from utils import get_data, preprocess_data, merge


def eval_RR_MSE():
    train_df, test_df = load_train_test()
    portfolio = Portfolio()
    portfolio.train(train_df)
    SE = 0
    accumulated = 1
    weighted_returns = []  # weighted return for each day in test
    RRs = []  # actual return for each stock and day in test
    RRs_hat = []  # predicted return for each stock and day in test
    for day in len(test_df):
        data = merge(train_df, test_df[:day])
        tmr_prices = portfolio.get_tomorrow_prices(data)
        tmr_portfolio = portfolio.get_portfolio(data)
        RR_hat = tmr_prices / test_df[day - 1]
        RR = test_df[day] / test_df[day - 1]
        weighted_returns.append(tmr_portfolio @ RR)
        RRs.append(RR)
        RRs_hat.append(RR_hat)

    weighted_returns = np.array(weighted_returns)
    RRs = np.array(RRs)
    RRs_hat = np.array(RRs_hat)

    SE = np.sum((RRs_hat - RRs) ** 2)
    MSE = SE / len(test_df)
    print(f"MSE of RR: {MSE}")

    MRR = np.mean(np.max(RRs, dim=1))
    MRR_hat = np.mean(weighted_returns)

    print(f"mean Max returns: {MRR}, mean weighted returns: {MRR_hat}")

    accumulated = np.prod(weighted_returns)
    max_accumulated = np.prod(np.max(RRs, dim=1))
    print(
        f"max accumulated returns: {max_accumulated}, accumulated returns: {accumulated}"
    )


def eval_portfolio_value():
    train_df, test_df = load_train_test()
    portfolio = Portfolio()
    portfolio.train(train_df)
    portfolio_value = 1
    for day in len(test_df):
        data = merge(train_df, test_df[:day])
        tmr = portfolio.get_portfolio(data)
        portfolio_value *= tmr / test_df[day - 1]


if __name__ == "__main__":
    main()
