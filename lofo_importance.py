import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm_notebook
import multiprocessing
from functools import partial


class LOFOImportance:

    def __init__(self, model, metric, df, features, target,
                 needs_proba=True, greater_is_better=True, num_folds=4):
        self.model = model
        self.metric = metric
        self.df = df
        self.features = features
        self.target = target
        self.needs_proba = needs_proba
        self.sign = 1 if greater_is_better else -1
        self.num_folds = num_folds

    def _calculate_performance(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train)
        if self.needs_proba:
            y_pred = self.model.predict_proba(X_val)[:, 1]
        else:
            y_pred = self.model.predict(X_val)

        return self.metric(y_val, y_pred)

    def _get_cv_score(self, feature_list):
        #print(feature_list)
        X = self.df[feature_list]
        y = self.df[self.target]
        
        importance = np.zeros(self.num_folds)
        kf = KFold(n_splits=self.num_folds, shuffle=True, random_state=0)

        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_val, y_val = X.iloc[val_index], y.iloc[val_index]

            importance[fold] = self._calculate_performance(X_train, y_train, X_val, y_val)

        return importance

    
    def get_importance(self):
        importances = np.zeros((len(self.features), self.num_folds))

        base_importance = self._get_cv_score(self.features)

        feature_lists = []
        for i, f in tqdm_notebook(enumerate(self.features)):
            feature_lists.append([feature for feature in self.features if feature != f])
        
        pool = multiprocessing.Pool(len(self.features)+1)
        importances = np.array(pool.map(partial(self._get_cv_score), feature_lists))
        
        importances = base_importance - importances
        importances *= self.sign
        
        importance_df = pd.DataFrame()
        importance_df["feature"] = self.features
        importance_df["importance_mean"] = importances.mean(axis=1)
        importance_df["importance_std"] = importances.std(axis=1)

        return importance_df.sort_values("importance_mean", ascending=False)
