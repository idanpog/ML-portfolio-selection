from main import test_portfolio
import numpy as np

def run(w, eps): -> float
from datetime import datetime, timedelta


def run(w, eps):
    # Fixed date: 2023-08-31
    today = datetime(2023, 8, 31)

    sharpe_results = []
    var_results = []

    for i in range(12):
        # Calculate the last day of the current month
        end_test_date = today.replace(day=1) - timedelta(days=1)
        # Calculate the day before the last day of the current month
        end_train_date = end_test_date - timedelta(days=1)
        # Calculate the start date, which is 5 years before the end_train_date
        start_date = end_train_date.replace(year=end_train_date.year - 5)

        # Call the test_portfolio function
        sharpe, var = test_portfolio(start_date, end_train_date, end_test_date)
        sharpe_results.append(sharpe)
        var_results.append(var)

        # Move to the previous month
        today = today.replace(day=1) - timedelta(days=1)
    return np.array(sharpe_results).mean()

