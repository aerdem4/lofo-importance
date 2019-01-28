import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from tqdm import tqdm_notebook
import multiprocessing
import warnings
from lofo.infer_defaults import infer_model


class LOFOImportance:

    def __init__(self, df, features, target,
                 scoring, model=None, cv=4, n_jobs=None):

        df = df.copy()
        self.fit_params = {}
        if model is None:
            model, df, categoricals = infer_model(df, features, target, n_jobs)
            self.fit_params["categorical_feature"] = categoricals
            n_jobs = 1

        self.model = model
        self.df = df
        self.features = features
        self.target = target
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        if self.n_jobs is not None and self.n_jobs > 1:
            warning_str = "Warning: If your model is multithreaded, please initialise the number \
                of jobs of LOFO to be equal to 1, otherwise you may experience issues."
            warnings.warn(warning_str)

    def _get_cv_score(self, X, y):
        fit_params = self.fit_params.copy()
        if "categorical_feature" in self.fit_params:
            fit_params["categorical_feature"] = [cat for cat in fit_params["categorical_feature"] if cat in X.columns]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = cross_validate(self.model, X, y, cv=self.cv, scoring=self.scoring, fit_params=fit_params)
        return cv_results['test_score']

    def _get_cv_score_parallel(self, feature, feature_list, result_queue, base=False):
        test_score = self._get_cv_score(self.df[feature_list], self.df[self.target])
        if not base:
            result_queue.put((feature, test_score))
        return test_score

    def get_importance(self):
        base_cv_score = self._get_cv_score(self.df[self.features], self.df[self.target])

        if self.n_jobs is not None and self.n_jobs > 1:
            pool = multiprocessing.Pool(self.n_jobs)
            manager = multiprocessing.Manager()
            result_queue = manager.Queue()

            base_cv_score = self._get_cv_score_parallel('all', self.features, result_queue, True)
            for f in self.features:
                feature_list = [feature for feature in self.features if feature != f]
                pool.apply_async(self._get_cv_score_parallel, (f, feature_list, result_queue))

            pool.close()
            pool.join()

            lofo_cv_scores = [result_queue.get() for _ in range(len(self.features))]
            lofo_cv_scores_normalized = np.array([base_cv_score - lofo_cv_score
                                                  for f, lofo_cv_score in lofo_cv_scores])
            self.features = [score[0] for score in lofo_cv_scores]
        else:
            lofo_cv_scores = []
            for i, f in tqdm_notebook(enumerate(self.features)):
                feature_list = [feature for feature in self.features if feature != f]
                lofo_cv_scores.append(self._get_cv_score(self.df[feature_list], self.df[self.target]))

            lofo_cv_scores_normalized = np.array([base_cv_score - lofo_cv_score for lofo_cv_score in lofo_cv_scores])

        importance_df = pd.DataFrame()
        importance_df["feature"] = self.features
        importance_df["importance_mean"] = lofo_cv_scores_normalized.mean(axis=1)
        importance_df["importance_std"] = lofo_cv_scores_normalized.std(axis=1)

        return importance_df.sort_values("importance_mean", ascending=False)
