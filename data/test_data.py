import numpy as np
import pandas as pd


def _to_binary(target):
    return (target > target.median()).astype(int)


def generate_test_data(data_size):
    df = pd.DataFrame()

    np.random.seed(0)
    df["A"] = np.random.rand(data_size)
    df["B"] = np.random.rand(data_size)
    df["C"] = np.random.rand(data_size)
    df["D"] = np.random.rand(data_size)

    df["target"] = 0.2 * np.random.rand(data_size) + df["A"] * df["D"] + 2 * df["B"]
    df["binary_target"] = _to_binary(df["target"])
    return df


def generate_unstructured_test_data(data_size):
    df = generate_test_data(data_size)
    df.loc[np.random.rand(data_size) < 0.3, "A"] = None
    df["E"] = np.random.choice(["category1", "category2", "category3"], data_size)

    df["target"] = (df["E"] != "category2")*df["target"]
    df["binary_target"] = _to_binary(df["target"])
    return df
