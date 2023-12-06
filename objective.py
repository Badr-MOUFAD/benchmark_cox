from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from skglm.datafits import Cox
    from skglm.penalties import L1_plus_L2
    from skglm.utils.jit_compilation import compiled_clone


class Objective(BaseObjective):

    name = "Cox Estimation"

    parameters = {
        'reg': [1e-1, 1e-2],
        'l1_ratio': [1., 0.7, 0.]
    }

    requirements = [
        "pip:git+https://github.com/scikit-learn-contrib/skglm.git@main",
    ]

    min_benchopt_version = "1.5"

    def __init__(self, reg=0.5, l1_ratio=1.):
        self.reg = reg
        self.l1_ratio = l1_ratio

    def set_data(self, X, y, use_efron):
        n_samples = X.shape[0]
        reg, l1_ratio = self.reg, self.l1_ratio

        self.X, self.y = X, y
        self.use_efron = use_efron

        # init penalty
        self.datafit = compiled_clone(Cox(self.use_efron))
        self.datafit.initialize(self.X, self.y)

        # init alpha
        grad_0 = self.datafit.raw_grad(self.y, np.zeros(n_samples))

        if l1_ratio != 0:
            self.alpha = reg * norm(X.T @ grad_0, ord=np.inf) / l1_ratio
        else:
            self.alpha = reg * norm(X.T @ grad_0, ord=np.inf)

        # init penalty
        self.penalty = compiled_clone(L1_plus_L2(self.alpha, self.l1_ratio))

    def evaluate_result(self, w):
        Xw = self.X @ w
        y = self.y

        datafit_val = self.datafit.value(y, w, Xw)
        penalty_val = self.penalty.value(w)

        return dict(
            value=datafit_val + penalty_val,
            support_size=(w != 0).sum(),
        )

    def get_objective(self):
        return dict(
            X=self.X, y=self.y,
            alpha=self.alpha, l1_ratio=self.l1_ratio,
            use_efron=self.use_efron
        )

    def get_one_result(self):
        n_features = self.X.shape[1]
        return dict(w=np.zeros(n_features))
