import numpy as np
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier, LGBMRegressor


def infer_model(df, features, y, n_jobs):
    model_class = LGBMRegressor
    if len(np.unique(y)) == 2:
        y = LabelEncoder().fit_transform(y)
        model_class = LGBMClassifier

    categoricals = []
    for f in features:
        if df[f].dtype == object:
            df[f] = LabelEncoder().fit_transform(df[f].apply(str))
            categoricals.append(f)

    min_child_samples = int(0.01*df.shape[0])

    model = model_class(min_child_samples=min_child_samples, n_jobs=n_jobs)

    return model, df, categoricals, y
