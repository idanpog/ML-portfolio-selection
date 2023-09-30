from main_new import test_portfolio
import numpy as np
from tqdm import tqdm
from datetime import datetime, timedelta

USE_PROPHET = False


def run(w, eps):
    # Fixed date: 2023-09-31
    today = datetime(2023, 8, 31)

    sharpe_results = []
    var_results = []

    for i in tqdm(range(8)):
        # Calculate the last day of the current month
        end_test_date = today.replace(day=1) - timedelta(days=1)

        # Calculate the end of the training period, which is one day before the start of the testing period
        end_train_date = end_test_date.replace(day=1) - timedelta(days=1)

        # Calculate the start date, which is 5 years before the end_train_date
        start_date = end_train_date.replace(year=end_train_date.year - 1) + timedelta(days=1)

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
    return np.array(sharpe_results).mean(), np.array(var_results).mean()


def run_omers_values():
    # Given data
    data = """
    2,"[0.1964909024089056, 0.0007755567002879046]","{'eps': 1.7870954024667873, 'w': 22}"
    7,"[0.20782752240969649, 0.0008010093526307387]","{'eps': 2.1580765548527094, 'w': 28}"
    9,"[0.15776005902464374, 0.0004483794097987892]","{'eps': 1.3148824239960164, 'w': 42}"
    10,"[0.19615987259695605, 0.000695117455735642]","{'eps': 1.6112266999729887, 'w': 31}"
    23,"[0.17878028837222842, 0.0005110313858780369]","{'eps': 1.341202801896298, 'w': 25}"
    30,"[0.18486361044682545, 0.0005404298079499216]","{'eps': 1.3853226827797278, 'w': 29}"
    36,"[0.15715442242615368, 0.00037134347290994016]","{'eps': 1.178530309202439, 'w': 20}"
    52,"[0.20857982734139846, 0.0010742829343057838]","{'eps': 2.897811214769126, 'w': 25}"
    """
    start_date = "2022-09-01"
    end_train_date = "2023-08-31"
    end_test_date = "2023-09-29"

    # Split the data into lines and then extract the parameters
    param_lines = data.strip().split("\n")
    params = []
    for line in param_lines:
        # Extracting the part which contains the parameters' dictionary
        param_str = line.split(',"{', 1)[-1].rsplit('}"', 1)[0]
        param_str = '{' + param_str + '}'
        param_str = param_str.replace("'", '"')  # Ensure all quotes are double quotes
        param_dict = eval(param_str)  # Convert the string representation to a dictionary
        params.append(param_dict)

    results = []
    for param in params:
        sharpe, var = test_portfolio(
            start_date,
            end_train_date,
            end_test_date,
            stg_params={"w": param['w'], "eps": param['eps'], "is_prophet": USE_PROPHET},
        )
        results.append({
            "w": param['w'],
            "eps": param['eps'],
            "sharpe": sharpe,
            "var": var
        })

    return results

#print(run_omers_values())
print(run(1,1))