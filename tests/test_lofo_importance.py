import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from lofo_importance import LOFOImportance

df = pd.DataFrame()

DATA_SIZE = 1000

np.random.seed(0)
df["A"] = np.random.rand(DATA_SIZE)
df["B"] = np.random.rand(DATA_SIZE)
df["C"] = np.random.rand(DATA_SIZE)
df["D"] = np.random.rand(DATA_SIZE)

df["target"] = 0.2*np.random.rand(DATA_SIZE) + df["A"]*df["D"] + 2*df["B"]
df["binary_target"] = (df["target"] > df["target"].median()).astype(int)
df.head()

lr = LinearRegression()
rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
cv = KFold(n_splits=4, shuffle=True, random_state=0)


def test_get_importance():
    fi = LOFOImportance(lr, df,
                        ["A", "B", "C", "D"], 'target',
                        cv=cv, scoring='neg_mean_absolute_error')

    importances = fi.get_importance()
    print(importances)
