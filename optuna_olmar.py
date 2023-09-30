"""
Author: Cfir Hadar

Description:
Parameters optimization for OLMAR algorithm using Optuna.
Parameters are: eps, w.

"""

import optuna
import cross_validate
import pandas as pd
from optuna.trial import TrialState
def objective(trial):
    # Define the search space
    eps = trial.suggest_float("eps", 1, 10)
    w = trial.suggest_int("w", 1, 60)

    # Call the cross_validate.run function
    return cross_validate.run(w, eps)


def main():
    # Maximize cross_validation.run function
    study = optuna.create_study(directions=["maximize",'minimize'])
    study.optimize(objective, timeout=10800,show_progress_bar=True)
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    complete_trials = [{'number': trial.number, 'value': trial.values, 'params': trial.params} for trial in complete_trials]
    # Save the DataFrame to a CSV file.
    df  = pd.DataFrame(complete_trials)
    df.to_csv("all_trials.csv", index=False)
    print(study.best_trials)
    # save the best parameters
    best_trials = study.best_trials
    best_trials = [{'number': trial.number, 'value': trial.values, 'params': trial.params} for trial in best_trials]
    df = pd.DataFrame(best_trials)
    df.to_csv("best_trials.csv", index=False)

if __name__ == "__main__":
    main()
