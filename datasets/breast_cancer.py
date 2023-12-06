from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sksurv.datasets import load_breast_cancer


class Dataset(BaseDataset):

    name = "breast"

    requirements = [
        "sebp::scikit-survival",
    ]

    parameters = {
        'normalize': [True],
        'with_ties': [True, False],
    }

    def __init__(self, normalize=True, with_ties=False):
        self.normalize = normalize
        self.with_ties = with_ties

    def get_data(self):
        X, y = load_breast_cancer()

        X = pd.get_dummies(
            X,
            drop_first=True,
            columns=X.select_dtypes(exclude="number").columns,
        )
        X = X.to_numpy(dtype=float)

        y = np.array(
            [[float(tm_i), float(s_i)] for s_i, tm_i in y],
            dtype=float,
            order='F'
        )

        if self.normalize:
            X = StandardScaler().fit_transform(X)

            tm = y[:, 0]
            tm = (tm - tm.mean()) / tm.std()
            tm -= tm.min()

            y[:, 0] = tm

        return dict(X=X, y=y, use_efron=self.with_ties)
