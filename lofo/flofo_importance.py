import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
import warnings
from sklearn.metrics import check_scoring
from lofo.utils import lofo_to_df, parallel_apply


class FLOFOImportance:
    """
    Fast LOFO Importance
    Applies already trained model on validation set by noising one feature each time.

    Parameters
    ----------
    trained_model: model (sklearn API)
        The model should be trained already
    validation_df: pandas dataframe
    features: list of strings
        List of column names for features within validation_df
    target: string
        Column name for target within validation_df
    scoring: string or callable
        Same as scoring in sklearn API
    n_jobs: int, optional
        Number of jobs for parallel computation
    """

    def __init__(self, trained_model, validation_df, features, target,
                 scoring, n_jobs=None):
        self.trained_model = trained_model
        self.df = validation_df.copy()
        self.features = features
        self.target = target
        self.n_jobs = n_jobs
        self.scorer = check_scoring(estimator=self.trained_model, scoring=scoring)

        # FLOFO defaults
        self.num_bins = 10
        self.shuffle_func = np.random.permutation
        self.feature_group_len = 2
        self.num_sampling = 10

        min_data_needed = 10*(self.num_bins**self.feature_group_len)
        if self.df.shape[0] < min_data_needed:
            raise Exception("Small validation set (<{})".format(min_data_needed))
        if len(self.features) <= self.feature_group_len:
            raise Exception("FLOFO needs more than {} features".format(self.feature_group_len))

        if self.n_jobs is not None and self.n_jobs > 1:
            warning_str = ("Warning: If your model is multithreaded, please initialise the number"
                           "of jobs of LOFO to be equal to 1, otherwise you may experience performance issues.")
            warnings.warn(warning_str)

        self._bin_features()

    def _bin_features(self):
        epsilon = 1e-10
        self.bin_df = pd.DataFrame()
        for feature in self.features:
            self.bin_df[feature] = self.df[feature].fillna(self.df[feature].median())
            self.bin_df[feature] = (self.bin_df[feature].rank(pct=True)*(self.num_bins - epsilon)).astype(np.int32)

    def _get_score(self, updated_df):
        return self.scorer(self.trained_model, updated_df[self.features], self.df[self.target])

    def _run(self, feature_name, n):
        scores = np.zeros(n)
        for i in range(n):
            feature_list = np.random.choice([feature for feature in self.features if feature != feature_name],
                                            size=self.feature_group_len, replace=False).tolist()
            self.bin_df["__f__"] = self.df[feature_name].values
            mutated_df = self.df.copy()
            mutated_df[feature_name] = self.bin_df.groupby(feature_list)["__f__"].transform(self.shuffle_func).values
            scores[i] = self._get_score(mutated_df)
        return scores

    def _run_parallel(self, feature_name, result_queue):
        test_score = self._run(feature_name, self.num_sampling)
        result_queue.put((feature_name, test_score))
        return test_score

    def get_importance(self, num_sampling=10, random_state=0):
        """Run FLOFO to get feature importances

        Parameters
        ----------
        num_sampling : int, optional
            Number of times features are shuffled
        random_state : int, optional
            Random seed

        Returns
        -------
        importance_df : pandas dataframe
            Dataframe with feature names and corresponding importance mean and std (sorted by importance)
        """
        np.random.seed(random_state)
        base_score = self._get_score(self.df)
        self.num_sampling = num_sampling

        if self.n_jobs is not None and self.n_jobs > 1:
            lofo_cv_scores = parallel_apply(self._run_parallel, self.features, self.n_jobs)
            lofo_cv_scores_normalized = np.array([base_score - lofo_cv_score for f, lofo_cv_score in lofo_cv_scores])
            self.features = [score[0] for score in lofo_cv_scores]
        else:
            lofo_cv_scores = []
            for f in tqdm(self.features):
                lofo_cv_scores.append(self._run(f, num_sampling))
            lofo_cv_scores_normalized = np.array([base_score - lofo_cv_score for lofo_cv_score in lofo_cv_scores])

        return lofo_to_df(lofo_cv_scores_normalized, self.features)
