from sklearn.ensemble import RandomForestClassifier
from lofo.lofo_importance import LOFOImportance
from data.test_data import generate_test_data


def test_lofo_importance():
    df = generate_test_data(1000)

    features = ["A", "B", "C", "D"]

    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)

    lofo = LOFOImportance(rf, df, features, 'binary_target', cv=4, scoring='roc_auc')

    importance_df = lofo.get_importance()

    assert len(features) == importance_df.shape[0], "Missing importance value for some features!"
    assert importance_df["feature"].values[0] == "B", "Most important feature is different than B!"
