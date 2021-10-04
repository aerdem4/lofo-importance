from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn.model_selection import KFold
from lofo import LOFOImportance, Dataset, plot_importance
from data.test_data import generate_test_data, generate_unstructured_test_data


def test_lofo_importance():
    df = generate_test_data(1000)
    features = ["A", "B", "C", "D"]
    dataset = Dataset(df=df, target="binary_target", features=features)

    lgbm = LGBMClassifier(random_state=0, n_jobs=4)

    lofo = LOFOImportance(dataset, model=lgbm, cv=4, scoring='roc_auc')

    importance_df = lofo.get_importance()

    plot_importance(importance_df)

    assert len(features) == importance_df.shape[0], "Missing importance value for some features!"
    assert importance_df["feature"].values[0] == "B", "Most important feature is different than B!"


def test_multithreading():
    df = generate_test_data(100000)
    features = ["A", "B", "C", "D"]
    dataset = Dataset(df=df, target="binary_target", features=features)

    lr = LogisticRegression(solver='liblinear')
    cv = KFold(n_splits=4, shuffle=True, random_state=0)

    lofo = LOFOImportance(dataset, model=lr, cv=cv, scoring='roc_auc', n_jobs=3)

    importance_df = lofo.get_importance()

    assert len(features) == importance_df.shape[0], "Missing importance value for some features!"
    assert importance_df["feature"].values[0] == "B", "Most important feature is different than B!"


def test_default_model():
    df = generate_unstructured_test_data(1000)
    features = ["A", "B", "C", "D", "E"]
    dataset = Dataset(df=df, target="target", features=features)

    lofo = LOFOImportance(dataset, cv=4, scoring='neg_mean_absolute_error')
    importance_df = lofo.get_importance()
    assert "E" in lofo.fit_params["categorical_feature"], "Categorical feature is not detected!"
    assert len(features) == importance_df.shape[0], "Missing importance value for some features!"

    df_checkpoint = df.copy()

    dataset = Dataset(df=df, target="binary_target", features=features)
    lofo = LOFOImportance(dataset, cv=4, scoring='roc_auc')
    importance_df = lofo.get_importance()

    assert "E" in lofo.fit_params["categorical_feature"], "Categorical feature is not detected!"
    assert df.equals(df_checkpoint), "LOFOImportance mutated the dataframe!"
    assert importance_df["feature"].values[0] == "E", "Most important feature is different than E!"


def test_feature_groups():
    df = generate_test_data(1000, text=True)
    features = ["A", "B", "C", "D"]

    cv = CountVectorizer(ngram_range=(3, 3), analyzer="char")
    feature_groups = dict()
    feature_groups["names"] = cv.fit_transform(df["T"])
    feature_groups["interactions"] = df[["A", "B"]].values*df[["C", "D"]].values

    dataset = Dataset(df=df, target="binary_target", features=features, feature_groups=feature_groups)

    lgbm = LGBMClassifier(random_state=0, n_jobs=4)

    lofo = LOFOImportance(dataset, model=lgbm, cv=4, scoring='roc_auc')

    importance_df = lofo.get_importance()

    assert len(features) + len(feature_groups) == importance_df.shape[0], "Missing importance value for some features!"
    assert importance_df["feature"].values[0] == "names", "Most important feature is different than 'names'!"
