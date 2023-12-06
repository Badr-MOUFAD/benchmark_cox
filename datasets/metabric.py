from benchopt import BaseDataset, safe_import_context

with safe_import_context() as import_ctx:
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer, make_column_transformer
    from sklearn.preprocessing import (
        StandardScaler, OneHotEncoder, OrdinalEncoder
    )


class Dataset(BaseDataset):

    name = "Breast-Cancer-Metabric"

    parameters = {
        'normalize': [True],
        "use_efron": [True, False]
    }

    def __init__(self, normalize=True, use_efron=True):
        self.normalize = normalize
        self.use_efron = use_efron

    def get_data(self):
        df = pd.read_csv("Breast Cancer METABRIC.csv")
        X, y = self.clean_data(df)

        if self.normalize:
            X = StandardScaler().fit_transform(X)

            tm = y[:, 0]
            tm = (tm - tm.mean()) / tm.std()
            tm -= tm[tm < 0].min()

            y[:, 0] = tm

        return dict(X=X, y=y, use_efron=self.use_efron)

    def clean_data(self, df):
        TARGET_COL = ["Overall Survival (Months)", "Overall Survival Status"]
        UNUSED_COL = [
            # redundant cols
            "Patient's Vital Status",
            "Relapse Free Status (Months)", "Relapse Free Status",
            # unique vals
            "Patient ID",
            # constant cols
            "Cancer Type", "Sex"
        ]

        df.drop(UNUSED_COL, axis=1, inplace=True)

        # handle missing values in y
        df.dropna(axis=0, subset=TARGET_COL, inplace=True)
        df = df.reindex()

        y = df[TARGET_COL]
        X = df.drop(TARGET_COL, axis=1)

        X_preprocessor = ColumnTransformer(
            [
                (
                    "numerical_prep",
                    SimpleImputer(strategy="mean"),
                    X.select_dtypes(include='number').columns
                ),
                (
                    "cat_prep",
                    OneHotEncoder(sparse=False, handle_unknown="ignore"),
                    X.select_dtypes(exclude='number').columns
                )
            ],
            remainder="passthrough"
        )
        y_preprocessor = make_column_transformer(
            (OrdinalEncoder(), ["Overall Survival Status"]),
            remainder='passthrough'
        )

        X = X_preprocessor.fit_transform(X)
        y = y_preprocessor.fit_transform(y)
        return X, y
