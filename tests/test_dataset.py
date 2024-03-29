import pytest
from data.test_data import generate_unstructured_test_data
from lofo import Dataset


def test_dataset():
    df = generate_unstructured_test_data(1000, text=True)
    features = ["A", "B", "C", "D", "D2", "E"]

    # Exception: feature group row count is not equal to the features' row count
    feature_groups = {"interactions": df[["A", "B"]].values[:10]*df[["C", "D"]].values[:10]}
    with pytest.raises(Exception):
        assert Dataset(df=df, target="binary_target", features=features, feature_groups=feature_groups)

    # Exception: Feature group name A is in use by other features
    feature_groups = {"A": df[["A", "B"]].values*df[["C", "D"]].values}
    with pytest.raises(Exception):
        assert Dataset(df=df, target="binary_target", features=features, feature_groups=feature_groups)

    # Exception: Feature group type is not numpy.ndarray or scipy.csr.csr_matrix
    feature_groups = {"F": df[["A", "B"]]}
    with pytest.raises(Exception):
        assert Dataset(df=df, target="binary_target", features=features, feature_groups=feature_groups)

    d = Dataset(df=df, target="binary_target", features=features, feature_groups={"F": df[["A", "B"]].values},
                auto_group_threshold=0.5)
    assert "D" not in d.feature_names and "D2" not in d.feature_names
    assert "D & D2" in d.feature_names and "F" in d.feature_groups.keys()
