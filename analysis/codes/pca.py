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
feat_dict = util.load_pickle('data/feat_dict_ready.pkl')

# %%

def gather_dict_values_to_list(dictionary):
    values = list(dictionary.values())
    l = []
    for i in values:

        for j in i:
            l.append(np.array(j))
    l = np.array(l)
    return l
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
<<<<<<< HEAD
# %%
=======

#%%
cumsum = np.cumsum(explained_variances)
i = 400
while cumsum[i] < 0.99:
    i += 1
print(i, cumsum[i])

#%%
def reverse_dict_values_to_list(feat_dict, feat_list):
    len_dict = {e1:len(e2) for e1, e2 in feat_dict.items()}
    pca_feats = {}
    i = 0
    for songurl, songlen in len_dict.items():
        pca_feats[songurl] = feat_list[i:i+songlen]
        i = i+songlen
        # print(i)
    # check
    temp = {e1:len(e2) for e1, e2 in pca_feats.items()}
    print(len_dict == temp)
    return pca_feats

#%%
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

train_dict = dict((songurl, feat_dict[songurl]) for songurl in util.trainlist)
test_dict = dict((songurl, feat_dict[songurl]) for songurl in util.testlist)

train_feats_list = gather_dict_values_to_list(train_dict)
test_feats_list = gather_dict_values_to_list(test_dict)

# Fit on training set only.
scaler.fit(train_feats_list)

# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_feats_list)
test_img = scaler.transform(test_feats_list)



#%%

from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(.99)
pca.fit(train_img)
train_pca = pca.transform(train_img)
test_pca = pca.transform(test_img)

#%%
train_data = reverse_dict_values_to_list(train_dict, train_pca)
test_data = reverse_dict_values_to_list(test_dict, test_pca)

#%%
import pickle

with open('data/train_feats_pca.pkl', 'wb') as handle:
    pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('data/test_feats_pca.pkl', 'wb') as handle:
    pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
>>>>>>> b85bc37c0deb81041273943d5bfa20000f3faf92
