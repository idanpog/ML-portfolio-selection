import pandas as pd
import yfinance as yf


def load_data(start_date="2015-1-1", end_date="2020-1-1", tickers=None):
    """
    Input: start_date: a string representing the start date of the data.
    end_date: a string representing the end date of the data.
    tickers: a list of strings representing the tickers of the stocks.
    Output: a dataframe containing the historical data of the stocks.
    """
    if not tickers:
        # use S&P500
        tickers = pd.read_html(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        tickers = tickers.Symbol.to_list()
    # Download the historical stock data for the tickers.
    data = yf.download(tickers, start=start_date, end=end_date).reset_index()
    data.rename(columns={'Date/': 'ds'}, inplace=True)
    
    closes = data.Close
    dates = data.Date

    data_dict = {tick: pd.concat([dates, pd.Series(closes[tick])], 1).rename(columns={'Date': 'ds', tick: 'y'}).fillna(0) for tick in tickers}

    return data_dict


def preprocess_data(data: pd.DataFrame):
    """replaces the stock values at each day with the relative return from the previous day
    Input: data: a dataframe containing the historical data of the stocks.
    """
    # Calculate the relative return of the stocks.
    data = data.pct_change()
    # Drop the first row of the dataframe.
    data = data.iloc[1:]
    # Return the dataframe.
    return data


def load_train_test(start_train, end_train, end_test):
    train = load_data(start_train, end_train)
    test = load_data(end_train, end_test)
    preprocess_data(train)
    preprocess_data(test)
    return train, test


def merge(data1, data2):
    return pd.concat([data1, data2], axis=0)
