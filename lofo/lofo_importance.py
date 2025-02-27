import numpy as np
from sklearn.model_selection import cross_validate
from tqdm.autonotebook import tqdm
import warnings
from lofo.infer_defaults import infer_model
from lofo.utils import lofo_to_df, parallel_apply
import sklearn


class LOFOImportance:
    """
    Leave One Feature Out Importance
    Given a model and cross-validation scheme, calculates the feature importances.

    Parameters
    ----------
    dataset: LOFO Dataset object
    scoring: string or callable
        Same as scoring in sklearn API
    model: model (sklearn API), optional
        Not trained model object
    fit_params : dict, optional
        fit parameters for the model
    cv: int or iterable
        Same as cv in sklearn API
    n_jobs: int, optional
        Number of jobs for parallel computation
    cv_groups: array-like, with shape (n_samples,), optional
        Group labels for the samples used while splitting the dataset into train/test set.
        Only used in conjunction with a “Group” cv instance (e.g., GroupKFold).
    """

    def __init__(self, dataset, scoring, model=None, fit_params=None, cv=4, n_jobs=None, cv_groups=None):

        self.fit_params = fit_params if fit_params else dict()
        if model is None:
            model, dataset.df, categoricals, dataset.y = infer_model(dataset.df, dataset.features, dataset.y, n_jobs)
            self.fit_params["categorical_feature"] = categoricals
            n_jobs = 1

        self.model = model
        self.dataset = dataset
        self.scoring = scoring
        self.cv = cv
        self.cv_groups = cv_groups
        self.n_jobs = n_jobs
        if self.n_jobs is not None and self.n_jobs > 1:
            warning_str = ("Warning: If your model is multithreaded, please initialise the number"
                           "of jobs of LOFO to be equal to 1, otherwise you may experience performance issues.")
            warnings.warn(warning_str)

        sklearn_version = tuple(map(int, sklearn.__version__.split(".")[:2]))
        self._cv_param_name = "params" if sklearn_version >= (1, 4) else "fit_params"

    def _get_cv_score(self, feature_to_remove):
        X, fit_params = self.dataset.getX(feature_to_remove=feature_to_remove, fit_params=self.fit_params)
        y = self.dataset.y

        kwargs = {self._cv_param_name: fit_params,
                  "cv": self.cv, "scoring": self.scoring, "groups": self.cv_groups}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = cross_validate(self.model, X, y, **kwargs)
        return cv_results['test_score']

    def _get_cv_score_parallel(self, feature, result_queue):
        test_score = self._get_cv_score(feature_to_remove=feature)
        result_queue.put((feature, test_score))
        return test_score

    def get_importance(self):
        """Run LOFO to get feature importances

        Returns
        -------
        importance_df : pandas dataframe
            Dataframe with feature names and corresponding importance mean and std (sorted by importance)
        """
        base_cv_score = self._get_cv_score(feature_to_remove=None)
        feature_list = self.dataset.feature_names + list(self.dataset.feature_groups.keys())

        if self.n_jobs is not None and self.n_jobs > 1:
            lofo_cv_result = parallel_apply(self._get_cv_score_parallel, feature_list, self.n_jobs)
            lofo_cv_scores_normalized = np.array([base_cv_score - lofo_cv_score for _, lofo_cv_score in lofo_cv_result])
            feature_list = [feature for feature, _ in lofo_cv_result]
        else:
            lofo_cv_scores = []
            for f in tqdm(feature_list):
                lofo_cv_scores.append(self._get_cv_score(feature_to_remove=f))
            lofo_cv_scores_normalized = np.array([base_cv_score - lofo_cv_score for lofo_cv_score in lofo_cv_scores])

        return lofo_to_df(lofo_cv_scores_normalized, feature_list)
