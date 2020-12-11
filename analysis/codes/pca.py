#%%

import numpy as np 
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
df.head()
#%%
df.iloc[:,[0,1,2,3]]
# %%
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(df.iloc[:,[0,1,2,3]])
X_scaled[:5]


# %%
features = X_scaled.T
cov_matrix = np.cov(features)
cov_matrix
# %%
values, vectors = np.linalg.eig(cov_matrix)
values
# %%
