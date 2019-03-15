from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from data.test_data import generate_test_data
from lofo.plotting import plot_importance
from lofo.flofo_importance import FLOFOImportance


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
