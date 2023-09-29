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
    study.optimize(objective, timeout=60,n_trials =40)
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    df = pd.DataFrame(complete_trials)

    # Save the DataFrame to a CSV file.
    df.to_csv("all_trials.csv", index=False)
    print(study.best_trial)
    # save the best parameters
    with open("optuna_olmar_best_params.txt", "w") as f:
        f.write(str(study.best_trials))


if __name__ == "__main__":
    main()
