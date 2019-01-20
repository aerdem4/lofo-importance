import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from tqdm import tqdm_notebook
import multiprocessing
import warnings


class LOFOImportance:

    def __init__(self, model, df, features, target,
                 scoring, cv=4, n_jobs=None):
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

    def _get_cv_score_singlethreaded(self, X, y):
        cv_results = cross_validate(self.model, X, y, cv=self.cv, scoring=self.scoring)
        return cv_results['test_score']

    def _get_cv_score_multithreaded(self, feature, feature_list, result_queue, base=False):
        cv_results = cross_validate(self.model, self.df[feature_list], self.df[self.target],
                                    cv=self.cv, scoring=self.scoring)
        if not base:
            result_queue.put({feature: cv_results['test_score']})
            return
        return cv_results['test_score']

    def get_importance(self):
        if self.n_jobs is not None and self.n_jobs > 1:
            pool = multiprocessing.Pool(self.n_jobs)
            manager = multiprocessing.Manager()
            result_queue = manager.Queue()

            base_cv_score = self._get_cv_score_multithreaded('all', self.features, result_queue, True)
            for f in self.features:
                feature_list = [feature for feature in self.features if feature != f]
                pool.apply_async(self._get_cv_score_multithreaded, (f, feature_list, result_queue))

            pool.close()
            pool.join()

            lofo_cv_scores = [result_queue.get() for _ in range(len(self.features))]
            lofo_cv_scores = {k: v for d in lofo_cv_scores for k, v in d.items()}
            lofo_cv_scores_normalized = np.array([base_cv_score - lofo_cv_score
                                                  for lofo_cv_score in lofo_cv_scores.values()])
            self.features = lofo_cv_scores.keys()
        else:
            base_cv_score = self._get_cv_score_singlethreaded(self.df[self.features], self.df[self.target])

            lofo_cv_scores = []
            for i, f in tqdm_notebook(enumerate(self.features)):
                feature_list = [feature for feature in self.features if feature != f]
                lofo_cv_scores.append(self._get_cv_score_singlethreaded(self.df[feature_list], self.df[self.target]))

            lofo_cv_scores_normalized = np.array([base_cv_score - lofo_cv_score for lofo_cv_score in lofo_cv_scores])

        importance_df = pd.DataFrame()
        importance_df["feature"] = self.features
        importance_df["importance_mean"] = lofo_cv_scores_normalized.mean(axis=1)
        importance_df["importance_std"] = lofo_cv_scores_normalized.std(axis=1)

        return importance_df.sort_values("importance_mean", ascending=False)
