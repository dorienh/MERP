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

import os
import glob
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
import util
# %%
## 1) load feat_dict
feat_dict = util.load_pickle('../../data/feat_dict_ready.pkl')
# %%
feats = list(feat_dict.values())
# %%
l = []
for i in feats:

    for j in i:
        l.append(np.array(j))
    # temp = np.array(temp)
    # l.append(temp)
l = np.array(l)
# %%
from sklearn.preprocessing import StandardScaler
X_scaled = StandardScaler().fit_transform(l)
# %%
features = X_scaled.T
cov_matrix = np.cov(features)
cov_matrix
# %%
values, vectors = np.linalg.eig(cov_matrix)
values
# %%
explained_variances = []
for i in range(len(values)):
    explained_variances.append(values[i] / np.sum(values))
 
print(np.sum(explained_variances[0:500]), '\n', explained_variances[0:500])
# %%
import matplotlib.pyplot as plt

plt.plot(np.cumsum(explained_variances))
# %%
