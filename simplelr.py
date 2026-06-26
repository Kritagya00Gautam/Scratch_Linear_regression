# %%
import pandas as pd
import numpy as np


# %%
teams = pd.read_csv("file")

# %%
X = teams[["atheletes",
           "prev_medals"]].copy()
y = teams[["medals"]].copy()


# %%
X["intercept"] = 1
X = X[["intercept", "atheletes", "prev_medals"]]


# %%
X_T = X.T


# %%
B = np.linalg.inv(X_T @ X) @ X_T @ y



# %%
B.index = X.columns

# %%
predictions = X @ B

# %%
SSR = ((y - predictions) ** 2).sum()
SST = ((y - y.mean()) ** 2 ).sum()
R2 =  1 - (SSR/SST)


