from benchopt import BaseObjective, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    from numpy.linalg import norm

    from skglm.datafits import Cox
    from skglm.penalties import L1_plus_L2
    from skglm.utils.jit_compilation import compiled_clone


class Objective(BaseObjective):

    name = "L2 Cox Estimation"

    parameters = {
        'reg': [1e-1, 1e-2],
    }

    # TODO: replace `pip:git+https://github.com/Badr-MOUFAD/skglm.git`
    # after merging skglm PR 159
    requirements = [
        "pip:git+https://github.com/scikit-learn-contrib/skglm.git@main",
    ]

    min_benchopt_version = "1.3"

    def __init__(self, reg=0.5):
        self.reg = reg

    def set_data(self, tm, s, X, use_efron):
        n_samples = X.shape[0]

        self.X = X
        self.y = (tm, s)
        self.use_efron = use_efron

        # init penalty
        self.datafit = compiled_clone(Cox(self.use_efron))
        self.datafit.initialize(self.X, self.y)

        # init alpha
        grad_0 = self.datafit.raw_grad(self.y, np.zeros(n_samples))
        self.alpha = self.reg * norm(X.T @ grad_0, ord=np.inf)

        # init penalty
        self.penalty = compiled_clone(L1_plus_L2(self.alpha, l1_ratio=0.))

    def compute(self, w):
        Xw = self.X @ w
        y = self.y

        datafit_val = self.datafit.value(y, w, Xw)
        penalty_val = self.penalty.value(w)

        return dict(
            value=datafit_val + penalty_val,
            support_size=(w != 0).sum(),
        )

    def get_objective(self):
        tm, s = self.y

        return dict(
            tm=tm, s=s, X=self.X, alpha=self.alpha,
            use_efron=self.use_efron
        )

    def get_one_solution(self):
        n_features = self.X.shape[1]
        return np.zeros(n_features)
