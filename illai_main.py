from ideas.prophet_portfolio import ProphetPortfolio
import utils
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
from prophet import Prophet

def split_by_date(df, date):
    return df[df.ds < date], df[df.ds >= date]

def calc_mean_ratio(ser1, ser2):
    return (ser1['y'].pct_change() / ser2['y'].pct_change()).mean()

def calc_rel_returns(ser):
    return ser['y'].pct_change()


if __name__ == '__main__':
    prophet_portfolio = ProphetPortfolio()
    # load data
    data = utils.load_data(start_date="2019-01-01", end_date="2020-09-30")

    aapl_split1, aapl_split2 = split_by_date(data['AAPL'], '2020-08-31')
    googl_split1, googl_split2 = split_by_date(data['GOOGL'], '2020-08-31')

    data_dict = {'AAPL': aapl_split1, 'GOOGL': googl_split1}
    # train model
    prophet_portfolio.train(data_dict)
    # predict
    aapl_preds = prophet_portfolio.predict_september("AAPL")[['yhat']].rename(columns={'yhat': 'y'})
    googl_preds = prophet_portfolio.predict_september("GOOGL")[['yhat']].rename(columns={'yhat': 'y'})

    real_ratio = calc_mean_ratio(aapl_split2, googl_split2)

    preds_ratio = calc_mean_ratio(aapl_preds, googl_preds)

    print(f"real ratio: {real_ratio}")
    print(f"preds ratio: {preds_ratio}")


    # tickers = ['GOOGL', 'AAPL']
    # newdata = yf.download(tickers, start="2019-01-01", end="2020-09-30").reset_index()

    # newdata.rename(columns={'Date/': 'ds'}, inplace=True)
    #
    # closes = newdata.Close
    # dates = newdata.Date
    #
    # data_dict = {
    #     tick: pd.concat([dates, pd.Series(closes[tick])], 1).rename(columns={'Date': 'ds', tick: 'y'}).fillna(0) for
    #     tick in tickers}

    # newmodel = Prophet().fit(data_dict['AAPL'])
    #
    # newmodel.plot(predictions, uncertainty=True)
    # plt.show()


