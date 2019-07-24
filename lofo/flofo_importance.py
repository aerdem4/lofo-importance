import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import multiprocessing
import warnings
from sklearn.metrics import check_scoring


class FLOFOImportance:

    def __init__(self, trained_model, validation_df, features, target,
                 scoring, n_jobs=None):
        self.trained_model = trained_model
        self.df = validation_df.copy()
        self.features = features
        self.target = target
        self.n_jobs = n_jobs

<<<<<<< HEAD
=======
        self.scorer = check_scoring(estimator=self.trained_model, scoring=scoring)

        # FLOFO defaults
        self.num_bins = 10
        self.shuffle_func = np.random.permutation
        self.feature_group_len = 2

        min_data_needed = 10*(self.num_bins**self.feature_group_len)
        if self.df.shape[0] < min_data_needed:
            raise Exception("Small validation set (<{})".format(min_data_needed))
        if len(self.features) <= self.feature_group_len:
            raise Exception("FLOFO needs more than {} features".format(self.feature_group_len))

>>>>>>> 9b186ba375193f6e73ae29bc41c2ad682d521919
        if self.n_jobs is not None and self.n_jobs > 1:
            warning_str = "Warning: If your model is multithreaded, please initialise the number \
                of jobs of LOFO to be equal to 1, otherwise you may experience issues."
            warnings.warn(warning_str)
<<<<<<< HEAD
        if self.df.shape[0] <= 1000:
            warnings.warn("Small validation set")

        self.scorer = check_scoring(estimator=self.trained_model, scoring=scoring)
        self.shuffle_func = np.random.permutation
        self._bin_features(10, 1e-10)

    def _bin_features(self, num_bins, epsilon):
        self.bin_df = pd.DataFrame()
        for feature in self.features:
            self.bin_df[feature] = self.df[feature].fillna(self.df[feature].median())
            self.bin_df[feature] = (self.bin_df[feature].rank(pct=True)*(num_bins - epsilon)).astype(np.int32)
=======

        self._bin_features()

    def _bin_features(self):
        epsilon = 1e-10
        self.bin_df = pd.DataFrame()
        for feature in self.features:
            self.bin_df[feature] = self.df[feature].fillna(self.df[feature].median())
            self.bin_df[feature] = (self.bin_df[feature].rank(pct=True)*(self.num_bins - epsilon)).astype(np.int32)
>>>>>>> 9b186ba375193f6e73ae29bc41c2ad682d521919

    def _get_score(self, updated_df):
        return self.scorer(self.trained_model, updated_df[self.features], self.df[self.target])

    def _run(self, feature_name, n):
        scores = np.zeros(n)
        for i in range(n):
            feature_list = np.random.choice([feature for feature in self.features if feature != feature_name],
<<<<<<< HEAD
                                            size=2, replace=False).tolist()
=======
                                            size=self.feature_group_len, replace=False).tolist()
>>>>>>> 9b186ba375193f6e73ae29bc41c2ad682d521919
            self.bin_df["__f__"] = self.df[feature_name].values
            mutated_df = self.df.copy()
            mutated_df[feature_name] = self.bin_df.groupby(feature_list)["__f__"].transform(self.shuffle_func).values
            scores[i] = self._get_score(mutated_df)
        return scores

    def _run_parallel(self, feature_name, n, result_queue):
        test_score = self._run(feature_name, n)
        result_queue.put((feature_name, test_score))
        return test_score

    def get_importance(self, num_sampling=10, random_state=0):
        np.random.seed(random_state)
        base_score = self._get_score(self.df)

        if self.n_jobs is not None and self.n_jobs > 1:
            pool = multiprocessing.Pool(self.n_jobs)
            manager = multiprocessing.Manager()
            result_queue = manager.Queue()

            for f in self.features:
                pool.apply_async(self._run_parallel, (f, num_sampling, result_queue))

            pool.close()
            pool.join()

            lofo_cv_scores = [result_queue.get() for _ in range(len(self.features))]
            lofo_cv_scores_normalized = np.array([base_score - lofo_cv_score for f, lofo_cv_score in lofo_cv_scores])
            self.features = [score[0] for score in lofo_cv_scores]
        else:
            lofo_cv_scores = []
            for f in tqdm_notebook(self.features):
                lofo_cv_scores.append(self._run(f, num_sampling))

            lofo_cv_scores_normalized = np.array([base_score - lofo_cv_score for lofo_cv_score in lofo_cv_scores])

        importance_df = pd.DataFrame()
        importance_df["feature"] = self.features
        importance_df["importance_mean"] = lofo_cv_scores_normalized.mean(axis=1)
        importance_df["importance_std"] = lofo_cv_scores_normalized.std(axis=1)

        return importance_df.sort_values("importance_mean", ascending=False)
