"""
Author: Cfir Hadar

Description:
Parameters optimization for OLMAR algorithm using Optuna.
Parameters are: eps, w.

"""

import optuna
import cross_validate


def main():
    # Maximize cross_validation.run function
    study = optuna.create_study(direction="maximize")
    study.optimize(cross_validate.run, n_trials=1000)
    print(study.best_trial)


if __name__ == "__main__":
    main()
