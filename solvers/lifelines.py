import warnings
from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion


with safe_import_context() as import_ctx:
    import pandas as pd
    import numpy as np
    from lifelines import CoxPHFitter


class Solver(BaseSolver):

    name = 'lifelines'

    requirements = [
        "lifelines",
    ]

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="iteration",
    )

    def set_objective(self, X, y, alpha, l1_ratio, use_efron):
        # format data
        stacked_y_X = np.hstack((y, X))
        self.df = pd.DataFrame(stacked_y_X)

        warnings.filterwarnings('ignore')
        self.estimator = CoxPHFitter(penalizer=alpha, l1_ratio=l1_ratio)

    def run(self, n_iter):
        self.estimator.fit(
            self.df,
            duration_col=0,
            event_col=1,
            fit_options={
                "max_steps": n_iter, "precision": 1e-12
            },
        )

        self.w = self.estimator.params_.values

    def get_result(self):
        return self.w

    def get_next(self, previous):
        "Linear growth for n_iter."
        return previous + 10
