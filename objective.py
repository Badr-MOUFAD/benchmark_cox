from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm


class Objective(BaseObjective):

    name = "L1 Cox Estimation"

    parameters = {
        'reg': [1e-1, 1e-2, 1e-3],
    }

    min_benchopt_version = "1.3"

    def __init__(self, reg):
        self.reg = reg

    def set_data(self, tm, s, X):
        self.tm, self.s, self.X = tm, s, X

        self.B = (tm >= tm[:, None]).astype(float)
        self.alpha = self.reg * Objective._compute_alpha_max(tm, s, X)

    def compute(self, w):
        s, X = self.s, self.X

        Xw = X @ w
        minus_log_lik = -(s @ Xw) + s @ np.log(self.B @ np.exp(Xw))
        penalty_val = self.alpha * norm(w, ord=1)

        return dict(
            value=minus_log_lik + penalty_val,
            support_size=(w != 0).sum(),
        )

    def get_one_solution(self):
        return np.zeros(self.X.shape[1])

    @staticmethod
    def _compute_alpha_max(tm, s, X):
        n_samples = X.shape[0]

        B = (tm >= tm[:, None]).astype(X.dtype)
        grad_0 = -s + B.T @ (s / np.sum(B, axis=1))

        return norm(X.T @ grad_0, ord=np.inf) / n_samples

    def get_objective(self):

        return dict(
            tm=self.tm, s=self.s, X=self.X, alpha=self.alpha,
        )
