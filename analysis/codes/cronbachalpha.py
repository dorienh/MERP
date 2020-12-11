#%% 

import pandas as pd
import numpy as np

# https://github.com/anthropedia/tci-stats/blob/master/tcistats/__init__.py
def cronbach_alpha(items):
    items = pd.DataFrame(items)
    items_count = items.shape[1]
    # print(items_count)
    variance_sum = float(items.var(axis=0, ddof=1).sum())
    total_var = float(items.sum(axis=1).var(ddof=1))
    
    return (items_count / float(items_count - 1) *
            (1 - variance_sum / total_var))



#%%
exps = pd.read_pickle(os.path.join(os.path.abspath('../..'), 'data', 'exps_ready.pkl'))
# %%
temp = exps.loc[exps['songurl']=='11_459']['arousals']
# %%
'checking for equal lengths between all participants'
[len(a) for a in temp]
# %%
# convert series of lists into a dataframe (matrix where x is time and y is participant)
tempdf = pd.DataFrame(temp.values.tolist())

# %%
cronbach_alpha(tempdf.transpose())
# %%

# find the cronbach alpha for all songs, then their mean and std.

datatype = 'valences'

cronalpha_list = []
cronalpha_dict = {}
for songurl, song_group in exps.groupby('songurl'):
    # if 'deam' not in songurl:
    #     pass
    # else:
    labels = song_group[datatype]
    labels_df = pd.DataFrame(labels.values.tolist()).transpose()
    # temp = [len(a) for a in labels]
    # print(all(x==temp[0] for x in temp))
    cronbach_alpha_value = cronbach_alpha(labels_df)
    cronalpha_list.append(cronbach_alpha_value)
    cronalpha_dict[songurl] = cronbach_alpha_value
print(cronalpha_dict)

print(f'{np.mean(cronalpha_list):5f} +- {np.std(cronalpha_list):5f}')



#%%
# EXTRA - find the total length of the 54 songs put together. min, max and mean lengths too.
import os
import sys
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
import util
import librosa

filepath = '/home/meowyan/Documents/emotion/data/50songs'
songfilenamedict = {}

y, sr = librosa.load(filename)

#%%