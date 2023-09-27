from ideas.prophet_portfolio import ProphetPortfolio
import utils

if __name__ == '__main__':
    prophet_portfolio = ProphetPortfolio()
    # load data
    data = utils.load_data(start_date="2015-01-01", end_date="2020-1-1")
    # train model
    prophet_portfolio.train(data)