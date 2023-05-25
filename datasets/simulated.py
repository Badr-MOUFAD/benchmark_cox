from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    from skglm.utils.data import make_dummy_survival_data


class Dataset(BaseDataset):

    name = "Simulated"

    requirements = [
        "git+https://github.com/Badr-MOUFAD/skglm.git@cox-estimator",
    ]

    parameters = {
        'n_samples, n_features': [
            (500, 500),
        ],
        'normalize': [True, False]
    }

    def __init__(self, n_samples=100, n_features=30,
                 normalize=True, random_state=1235):
        self.n_samples = n_samples
        self.n_features = n_features
        self.random_state = random_state
        self.normalize = normalize

    def get_data(self):
        tm, s, X = make_dummy_survival_data(
            self.n_samples, self.n_features, self.normalize,
            random_state=self.random_state)

        return dict(tm=tm, s=s, X=X)
