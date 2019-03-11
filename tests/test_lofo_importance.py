from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold, train_test_split
from lofo.lofo_importance import LOFOImportance
from lofo.flofo_importance import FLOFOImportance
from lofo.plotting import plot_importance
from data.test_data import generate_test_data, generate_unstructured_test_data


def test_lofo_importance():
    df = generate_test_data(1000)

    features = ["A", "B", "C", "D"]

    lgbm = LGBMClassifier(random_state=0, n_jobs=4)

    lofo = LOFOImportance(df, features, 'binary_target', model=lgbm, cv=4, scoring='roc_auc')

    importance_df = lofo.get_importance()

    plot_importance(importance_df)

    assert len(features) == importance_df.shape[0], "Missing importance value for some features!"
    assert importance_df["feature"].values[0] == "B", "Most important feature is different than B!"


def test_multithreading():
    df = generate_test_data(100000)

    features = ["A", "B", "C", "D"]

    lr = LogisticRegression(solver='liblinear')
    cv = KFold(n_splits=4, shuffle=True, random_state=0)

    lofo = LOFOImportance(df, features, 'binary_target', model=lr, cv=cv, scoring='roc_auc', n_jobs=3)

    importance_df = lofo.get_importance()

    assert len(features) == importance_df.shape[0], "Missing importance value for some features!"
    assert importance_df["feature"].values[0] == "B", "Most important feature is different than B!"


def test_default_model():
    df = generate_unstructured_test_data(1000)
    df_checkpoint = df.copy()

    features = ["A", "B", "C", "D", "E"]

    lofo = LOFOImportance(df, features, 'target', cv=4, scoring='neg_mean_absolute_error')
    importance_df = lofo.get_importance()
    assert len(features) == importance_df.shape[0], "Missing importance value for some features!"

    lofo = LOFOImportance(df, features, 'binary_target', cv=4, scoring='roc_auc')
    importance_df = lofo.get_importance()

    assert df.equals(df_checkpoint), "LOFOImportance mutated the dataframe!"
    assert importance_df["feature"].values[0] == "E", "Most important feature is different than E!"


def test_flofo_importance():
    df = generate_test_data(100000)
    df.loc[df["A"] < df["A"].median(), "A"] = None

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=0)
    val_df_checkpoint = val_df.copy()

    features = ["A", "B", "C", "D"]

    lgbm = LGBMClassifier(random_state=0, n_jobs=1)
    lgbm.fit(train_df[features], train_df["binary_target"])

    flofo = FLOFOImportance(lgbm, df, features, 'binary_target', scoring='roc_auc')
    flofo_parallel = FLOFOImportance(lgbm, df, features, 'binary_target', scoring='roc_auc', n_jobs=3)

    importance_df = flofo.get_importance()
    importance_df_parallel = flofo_parallel.get_importance()
    is_feature_order_same = importance_df["feature"].values == importance_df_parallel["feature"].values

    plot_importance(importance_df)

    assert is_feature_order_same.sum() == len(features), "Parallel FLOFO returned different result!"
    assert val_df.equals(val_df_checkpoint), "LOFOImportance mutated the dataframe!"
    assert len(features) == importance_df.shape[0], "Missing importance value for some features!"
    assert importance_df["feature"].values[0] == "B", "Most important feature is different than B!"
