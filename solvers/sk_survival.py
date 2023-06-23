import warnings
from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

with safe_import_context() as import_ctx:
    import numpy as np
    from sksurv.linear_model import (CoxnetSurvivalAnalysis,
                                     CoxPHSurvivalAnalysis)


class Solver(BaseSolver):

    name = 'scikit-survival'

    requirements = [
        "sebp::scikit-survival",
    ]

    reference = [
        "S. Pölsterl, 'scikit-survival: A Library for Time-to-Event Analysis "
        "Built on Top of scikit-learn.', Journal of Machine"
        "Learning Research, vol. 21, no. 212, pp. 1-6, 2020."
    ]

    stopping_strategy = 'iteration'

    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="iteration",
    )

    def set_objective(self, X, y, alpha, l1_ratio, use_efron):
        self.X, self.y = X, y
        self.l1_ratio = l1_ratio
        n_samples = X.shape[0]

        # cast data
        dtype = np.dtype([('fstat', bool), ('lenfol', '<f8')])
        tm, s = y[:, 0], y[:, 1]
        self.y = np.array(list(
            zip(s, tm)
        ), dtype)

        warnings.filterwarnings('ignore')
        if l1_ratio != 0:
            self.estimator = CoxnetSurvivalAnalysis(
                alphas=[alpha],
                l1_ratio=l1_ratio,
                tol=1e-12
            )
        else:
            self.estimator = CoxPHSurvivalAnalysis(
                alpha=n_samples * alpha,
                ties='efron' if use_efron else 'breslow',
                tol=1e-12,
            )

        # cache potential compilation
        self.run(4)

    def run(self, n_iter):
        if n_iter == 0:
            self.w = np.zeros(self.X.shape[1])
            return

        if isinstance(self.estimator, CoxnetSurvivalAnalysis):
            self.estimator.max_iter = n_iter
        else:
            self.estimator.n_iter = n_iter

        self.estimator.fit(self.X, self.y)
        self.w = self.estimator.coef_

    def get_result(self):
        return self.w.flatten()

    def skip(self, X, y, alpha, l1_ratio, use_efron):
        if l1_ratio != 0 and use_efron:
            reason = (f"{self.name} does not handle tied data"
                      " for Elastic Cox estimation.")
            return True, reason

        return False, None

    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 10
