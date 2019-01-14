import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from tqdm import tqdm_notebook
import multiprocessing
from functools import partial


class LOFOImportance:

    def __init__(self, model, df, features, target,
                 scoring, cv=4):
        self.model = model
        self.df = df
        self.features = features
        self.target = target
        self.scoring = scoring
        self.cv = cv

    def _get_cv_score(self, X, y):
        cv_results = cross_validate(self.model, X, y, cv=self.cv, scoring=self.scoring)
        return cv_results['test_score']

    
    def get_importance(self):
        lofo_cv_scores = []

        base_cv_score = self._get_cv_score(self.df[self.features], self.df[self.target])

        feature_lists = []
        for i, f in tqdm_notebook(enumerate(self.features)):
            feature_list = [feature for feature in self.features if feature != f]
            lofo_cv_scores.append(self._get_cv_score(self.df[feature_list], self.df[self.target]))

        lofo_cv_scores_normalized = np.array([base_cv_score - lofo_cv_score for lofo_cv_score in lofo_cv_scores])

        importance_df = pd.DataFrame()
        importance_df["feature"] = self.features
        importance_df["importance_mean"] = lofo_cv_scores_normalized.mean(axis=1)
        importance_df["importance_std"] = lofo_cv_scores_normalized.std(axis=1)

        return importance_df.sort_values("importance_mean", ascending=False)
