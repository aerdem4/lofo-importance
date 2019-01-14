import numpy as np
import pandas as pd


def generate_test_data(data_size):
    df = pd.DataFrame()

    np.random.seed(0)
    df["A"] = np.random.rand(data_size)
    df["B"] = np.random.rand(data_size)
    df["C"] = np.random.rand(data_size)
    df["D"] = np.random.rand(data_size)

    df["target"] = 0.2 * np.random.rand(data_size) + df["A"] * df["D"] + 2 * df["B"]
    df["binary_target"] = (df["target"] > df["target"].median()).astype(int)
    return df
