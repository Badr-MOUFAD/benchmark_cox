import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from sksurv.linear_model import CoxnetSurvivalAnalysis


class Solver(BaseSolver):

    name = 'scikit-survival'

    requirements = [
        "sebp::scikit-survival",
    ]

    stopping_strategy = 'iteration'

    def set_objective(self, tm, s, X, alpha, l1_ratio, use_efron):
        self.tm, self.s, self.X = tm, s, X
        self.l1_ratio = l1_ratio

        # cast data
        dtype = np.dtype([('fstat', bool), ('lenfol', '<f8')])
        self.y = np.array(list(
            zip(s, tm)
        ), dtype)

        warnings.filterwarnings('ignore')
        self.estimator = CoxnetSurvivalAnalysis(
            alphas=[alpha],
            l1_ratio=l1_ratio,
            tol=1e-12
        )

        # cache potential compilation
        self.run(4)

    def run(self, n_iter):
        if n_iter == 0:
            self.w = np.zeros(self.X.shape[1])
            return

        self.estimator.max_iter = n_iter

        self.estimator.fit(self.X, self.y)
        self.w = self.estimator.coef_

    def get_result(self):
        return self.w.flatten()

    def skip(self, tm, s, X, alpha, l1_ratio, use_efron):
        if use_efron:
            reason = (f"{self.name} does not handle tied data"
                      " for penalized Cox estimation.")
            return True, reason

        return False, None

    @staticmethod
    def get_next(previous):
        "Linear growth for n_iter."
        return previous + 50
