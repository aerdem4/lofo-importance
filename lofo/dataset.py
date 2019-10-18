import numpy as np
import scipy.sparse as ss


class Dataset:
    """
    Dataset for LOFO

    Parameters
    ----------
    df: pandas dataframe
    target: string
        Column name for target within df
    features: list of strings
        List of column names within df
    feature_groups: dict, optional
        Name, value dictionary of feature groups as numpy.darray or scipy.csr.scr_matrix
    """

    def __init__(self, df, target, features, feature_groups=None):
        self.df = df.copy()
        self.features = list(features)
        self.feature_groups = feature_groups if feature_groups else dict()

        self.num_rows = df.shape[0]
        self.y = df[target].values

        for feature_name, feature_matrix in self.feature_groups.items():
            if not (isinstance(feature_matrix, np.ndarray) or isinstance(feature_matrix, ss.csr.csr_matrix)):
                raise Exception("Data type {dtype} is not a valid type!".format(dtype=type(feature_matrix)))

            if feature_matrix.shape[0] != self.num_rows:
                raise Exception("Expected {expected} rows but got {n} rows!".format(expected=self.num_rows,
                                                                                    n=feature_matrix.shape[0]))

            if feature_name in self.features:
                raise Exception("Feature group name '{name}' is the same with one of the features!")

    def getX(self, feature_to_remove, fit_params):
        """Get feature matrix and fit_params after removing a feature

        Parameters
        ----------
        feature_to_remove : string
            feature name to remove
        fit_params : dict
            fit parameters for the model

        Returns
        -------
        X : numpy.darray or scipy.csr.scr_matrix
            Feature matrix
        fit_params: dict
            Updated fit_params after feature removal
        """
        feature_list = [feature for feature in self.features if feature != feature_to_remove]
        concat_list = [self.df[feature_list].values]

        for feature_name, feature_matrix in self.feature_groups.items():
            if feature_name != feature_to_remove:
                concat_list.append(feature_matrix)

        fit_params = fit_params.copy()
        if "categorical_feature" in fit_params:
            cat_features = [f for f in fit_params["categorical_feature"] if f != feature_to_remove]
            fit_params["categorical_feature"] = [ix for ix, f in enumerate(feature_list) if (f in cat_features)]

        has_sparse = False
        for feature_name, feature_matrix in self.feature_groups.items():
            if feature_name != feature_to_remove and isinstance(feature_matrix, ss.csr.csr_matrix):
                has_sparse = True

        concat = np.hstack
        if has_sparse:
            concat = ss.hstack

        return concat(concat_list), fit_params
