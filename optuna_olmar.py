"""
Author: Cfir Hadar

Description:
Parameters optimization for OLMAR algorithm using Optuna.
Parameters are: eps, w.

"""

import optuna
import cross_validate


def objective(trial):
    # Define the search space
    eps = trial.suggest_float("eps", 1, 10)
    w = trial.suggest_int("w", 1, 200)

    # Call the cross_validate.run function
    return cross_validate.run(w, eps)


def main():
    # Maximize cross_validation.run function
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=120)
    print(study.best_trial)
    # save the best parameters
    with open("optuna_olmar_best_params.txt", "w") as f:
        f.write(str(study.best_trial))


if __name__ == "__main__":
    main()
