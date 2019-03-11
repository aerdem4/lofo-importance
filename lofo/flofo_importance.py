import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate
from tqdm import tqdm_notebook
import multiprocessing
import warnings
from lofo.infer_defaults import infer_model


class FLOFOImportance:

    def __init__(self, predict_func, df, features, target,
                 scoring, n_jobs=None):

        df = df.copy()
        self.predict_func = predict_func
        self.df = df
        self.features = features
        self.target = target
        self.scoring = scoring
        self.n_jobs = n_jobs
        if self.n_jobs is not None and self.n_jobs > 1:
            warning_str = "Warning: If your model is multithreaded, please initialise the number \
                of jobs of LOFO to be equal to 1, otherwise you may experience issues."
            warnings.warn(warning_str)
        if df.shape[0] <= 1000:
            warnings.warn("Small validation set")

        self._bin_features()

    def _bin_features(self):
        self.bin_df = pd.DataFrame()
        for feature in self.features:
            self.bin_df[feature] = (self.df[feature].rank(pct=True)*(10 - 0.001)).astype(int)

    def _get_score(self, df):
        return self.scoring._sign*self.scoring._score_func(self.predict_func(df[self.features]), self.df[self.target])

    def _run(self, feature_name, n):
        scores = np.zeros(n)
        for i in range(n):
            feature_list = np.random.choice([feature for feature in self.features if feature != feature_name],
                                            size=2, replace=False).tolist()
            self.bin_df["__f__"] = self.df[feature_name].values
            mutated_df = self.df.copy()
            mutated_df[feature_name] = self.bin_df.groupby(feature_list)["__f__"].transform(np.random.permutation).values
            scores[i] = self._get_score(mutated_df)
        return scores

    def _run_parallel(self, feature_name, n, result_queue):
        test_score = self._run(feature_name, n)
        result_queue.put((feature_name, test_score))
        return test_score

    def get_importance(self):
        base_score = self._get_score(self.df)
        n = 10

        if self.n_jobs is not None and self.n_jobs > 1:
            pool = multiprocessing.Pool(self.n_jobs)
            manager = multiprocessing.Manager()
            result_queue = manager.Queue()

            for f in self.features:
                pool.apply_async(self._run_parallel, (f, n, result_queue))

            pool.close()
            pool.join()

            lofo_cv_scores = [result_queue.get() for _ in range(len(self.features))]
            lofo_cv_scores_normalized = np.array([base_score - lofo_cv_score for f, lofo_cv_score in lofo_cv_scores])
            self.features = [score[0] for score in lofo_cv_scores]
        else:
            lofo_cv_scores = []
            for f in tqdm_notebook(self.features):
                lofo_cv_scores.append(self._run(f, n))

            lofo_cv_scores_normalized = np.array([base_score - lofo_cv_score for lofo_cv_score in lofo_cv_scores])

        importance_df = pd.DataFrame()
        importance_df["feature"] = self.features
        importance_df["importance_mean"] = lofo_cv_scores_normalized.mean(axis=1)
        importance_df["importance_std"] = lofo_cv_scores_normalized.std(axis=1)

        return importance_df.sort_values("importance_mean", ascending=False)


def plot_importance(importance_df, figsize=(8, 8)):
    importance_df = importance_df.copy()
    importance_df["color"] = (importance_df["importance_mean"] > 0).map({True: 'g', False: 'r'})
    importance_df.sort_values("importance_mean", inplace=True)

    importance_df.plot(x="feature", y="importance_mean", xerr="importance_std",
                       kind='barh', color=importance_df["color"], figsize=figsize)
