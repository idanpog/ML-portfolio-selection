from main import test_portfolio
import numpy as np

from datetime import datetime, timedelta

USE_PROPHET = False


def run(w, eps):
    # Fixed date: 2023-09-31
    today = datetime(2023, 8, 31)

    sharpe_results = []
    var_results = []

    for i in range(12):
        # Calculate the last day of the current month
        end_test_date = today.replace(day=1) - timedelta(days=1)

        # Calculate the end of the training period, which is one day before the start of the testing period
        end_train_date = end_test_date.replace(day=1) - timedelta(days=1)

        # Calculate the start date, which is 5 years before the end_train_date
        start_date = end_train_date.replace(year=end_train_date.year - 4) + timedelta(days=1)

        # Call the test_portfolio function
        sharpe, var = test_portfolio(
            start_date,
            end_train_date,
            end_test_date,
            stg_params={"w": w, "eps": eps, "is_prophet": USE_PROPHET},
        )
        sharpe_results.append(sharpe)
        var_results.append(var)

        # Move to the previous month
        today = today.replace(day=1) - timedelta(days=1)
    return np.array(sharpe_results).mean(),np.array(var_results).mean()

run(50, 2)