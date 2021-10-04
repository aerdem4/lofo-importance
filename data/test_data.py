import numpy as np
import pandas as pd


def _to_binary(target):
    return (target > target.median()).astype(int)


def generate_test_data(data_size, text=False):
    df = pd.DataFrame()

    np.random.seed(0)
    df["A"] = np.random.rand(data_size)
    df["B"] = np.random.rand(data_size)
    df["C"] = np.random.rand(data_size)
    df["D"] = np.random.rand(data_size)

    df["D2"] = df["D"].values + 0.1*np.random.rand(data_size)
    df.loc[df["D2"] > 1, "D2"] = None

    df["target"] = 0.2 * np.random.rand(data_size) + df["A"] * df["D"] + 2 * df["B"]
    df["binary_target"] = _to_binary(df["target"])

    if text:
        df["T"] = np.random.choice(["Bojack", "Horseman", "Todd", "Chavez"], data_size)
        df["target"] *= (df["T"] == "Todd")
        df["binary_target"] *= (df["T"] == "Todd")

    return df


def generate_unstructured_test_data(data_size, text=False):
    df = generate_test_data(data_size, text)
    df.loc[np.random.rand(data_size) < 0.3, "A"] = None
    df["E"] = np.random.choice(["category1", "category2", "category3"], data_size)
    df["E"] = df["E"].astype("category")

    df["target"] = (df["E"] != "category2")*df["target"]
    df["binary_target"] = _to_binary(df["target"])
    return df
