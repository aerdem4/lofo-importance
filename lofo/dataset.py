import itertools
import numpy as np
import scipy.sparse as ss
from scipy.stats import spearmanr
from lofo.utils import flatten_list
import networkx as nx


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
    auto_group_threshold: float, optional
        Threshold for grouping correlated features together, must be between 0 and 1
    """

    def __init__(self, df, target, features, feature_groups=None, auto_group_threshold=1.0):
        self.df = df.copy()
        self.features = list(features)
        self.feature_groups = feature_groups if feature_groups else dict()

        self.num_rows = df.shape[0]
        self.target_name = target
        self.y = df[self.target_name].values

        grouped_features, auto_groups = self.auto_group_features(auto_group_threshold)
        self.features = [[f] for f in list(set(self.features) - set(grouped_features))] + auto_groups
        self.feature_names = [" & ".join(feature_list) for feature_list in self.features]

        if len(auto_groups) > 0:
            print("Automatically grouped features by correlation:")
            for i in range(len(auto_groups)):
                print(i + 1, auto_groups[i])

        for feature_name, feature_matrix in self.feature_groups.items():
            if not (isinstance(feature_matrix, np.ndarray) or isinstance(feature_matrix, ss.csr.csr_matrix)):
                raise Exception("Data type {dtype} is not a valid type!".format(dtype=type(feature_matrix)))

            if feature_matrix.shape[0] != self.num_rows:
                raise Exception("Expected {expected} rows but got {n} rows!".format(expected=self.num_rows,
                                                                                    n=feature_matrix.shape[0]))

            if feature_name in self.feature_names:
                same_name_exception = "Feature group name '{name}' is the same with one of the features!"
                raise Exception(same_name_exception.format(name=feature_name))

    def auto_group_features(self, auto_group_threshold):
        if auto_group_threshold == 1.0:
            return [], []
        elif auto_group_threshold == 0.0:
            grouped_features = list(self.features)
            auto_groups = [set(self.features)]
            return grouped_features, auto_groups
        elif 0 < auto_group_threshold < 1:
            feature_matrix = self.df[self.features].values

            for i, feature in enumerate(self.features):
                if self.df[feature].dtype.name == "category":
                    feature_series = self.df.groupby(feature)[self.target_name].transform("mean")
                else:
                    feature_series = self.df[feature]
                feature_matrix[:, i] = feature_series.fillna(feature_series.mean()).fillna(0).values

            corr_matrix, _ = spearmanr(np.nan_to_num(feature_matrix))
            corr_matrix = np.abs(corr_matrix)

            G = nx.Graph()

            for i in range(len(self.features)):
                for j in range(i + 1, len(self.features)):
                    if corr_matrix[i, j] > auto_group_threshold:
                        G.add_edge(i, j)

            subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]

            groups = []
            for sg in subgraphs:
                groups.append([self.features[node] for node in sg.nodes()])

            auto_groups = [sorted(g) for g in groups]
            grouped_features = list(itertools.chain(*[list(g) for g in groups]))
            return grouped_features, auto_groups
        else:
            raise Exception("auto_group_threshold must be between 0 and 1 (inclusive)!")

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
        feature_lists = [self.features[i] for i, feature_name in enumerate(self.feature_names)
                         if feature_name != feature_to_remove]
        feature_list = flatten_list(feature_lists)
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
