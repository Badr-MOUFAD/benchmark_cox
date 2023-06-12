import warnings
from benchopt import BaseSolver, safe_import_context

with safe_import_context() as import_ctx:
    from skglm.datafits import Cox
    from skglm.penalties import L1_plus_L2
    from skglm.solvers import ProxNewton
    from skglm.utils.jit_compilation import compiled_clone


class Solver(BaseSolver):

    name = 'skglm'

    # TODO: replace `pip:git+https://github.com/Badr-MOUFAD/skglm.git`
    # after merging skglm PR 159
    requirements = [
        "pip:git+https://github.com/Badr-MOUFAD/skglm.git@cox-efron",
    ]

    stopping_strategy = 'iteration'

    def set_objective(self, tm, s, X, alpha, use_efron):
        self.tm, self.s, self.X = tm, s, X

        # fit ProxNewton
        self.datafit = compiled_clone(Cox(use_efron))
        self.penalty = compiled_clone(L1_plus_L2(alpha, l1_ratio=0.))

        self.datafit.initialize(X, (tm, s))

        warnings.filterwarnings('ignore')
        self.solver = ProxNewton(fit_intercept=False, tol=1e-9)

        # cache numba compilation
        self.run(4)

    def run(self, n_iter):
        self.solver.max_iter = n_iter

        w, *_ = self.solver.solve(
            self.X, (self.tm, self.s), self.datafit, self.penalty
        )

        self.w = w

    def get_result(self):
        return self.w
