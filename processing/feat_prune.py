'''
1) Remove the fist 15 seconds of audio features and participant given labels
2) Resample to 0.5 seconds pre label and per feature.
'''

#%%
import os
import sys
sys.path.append(os.path.abspath('..'))
print(sys.path)
import pickle
import numpy as np
import pandas as pd

import util

#%%
# load data
feat_dict = util.load_pickle('../data/feat_dict.pkl')
# exps = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'exps_rescaled_smoothed.pkl'))

# %%

'''
https://stackoverflow.com/questions/10847660/subsampling-averaging-over-a-numpy-array
the last few datapoints that don't make up 0.5 seconds are thrown away.
'''
def average_2D(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, 1582, n), 2) # 1582 is the number of types of features extracted in opensmile

def average_1D(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


#%%
###################################################
# feat_dict: remove head
###################################################
'''
After extracting features from opensmile,
odd number of seconds will result in 12 less datapoints than expected (even number of datapoints)
even number of seconds will results in 11 less datapoints than expected (odd number of datapoints)
but we shall ignore that fact and simply remove 150 datapoints from the front. 
'''
head_cutoff = 150

def check_lens_feat_dict(feat_dict):
    len_dict = {}
    for key, value in feat_dict.items():
        len_dict[key] = len(value)
    print(len_dict)

# %%
feat_dict_headless = {}
for key, value in feat_dict.items():
    feat_dict_headless[key] = value[head_cutoff:]

# print both lengths to check~
check_lens_feat_dict(feat_dict)
check_lens_feat_dict(feat_dict_headless)

#%%
###################################################
# exps: remove head
'''SHIFTED TO pruning_deam.py'''
###################################################


# for rowidx in exps.index:
#     exps.at[rowidx, 'arousals'] = exps.at[rowidx, 'arousals'][head_cutoff:]
#     exps.at[rowidx, 'valences'] = exps.at[rowidx, 'valences'][head_cutoff:]

# temp = exps.loc[0]
# print(np.shape(temp['arousals']))
# print(temp['songurl'])
# count = 0
# for idx in exps.index:
#     row = exps.loc[idx]
#     count += len(row['arousals'])
# print(count)




# %%
###################################################
# feat_dict: rescale from 0.1s to 0.5s per label
###################################################

sample_factor = 5

#%%
## print stuff to check
# temp = exps.loc[0]
# songurl = temp['songurl']
# print('songurl: ', songurl)
# print('feature shape: ', np.shape(feat_dict[songurl]))
# print('a label shape: ', np.shape(temp['arousals']))
# print('v label shape: ', np.shape(temp['valences']))


# %%
'''
https://stackoverflow.com/questions/10847660/subsampling-averaging-over-a-numpy-array
the last few datapoints that don't make up 0.5 seconds are thrown away.
'''
def average_2D(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, 1582, n), 2) # 1582 is the number of types of features extracted in opensmile

def average_1D(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)


# %%

temp = feat_dict_headless['10_828']
print(np.shape(temp))

feat_dict_headless_rescaled = {}

for key, value in feat_dict_headless.items():
    feat_dict_headless_rescaled[key] = average_2D(value, sample_factor)

temp = feat_dict_headless_rescaled['10_828']
print(np.shape(temp))


#%%
###################################################
# NORMALIZE FEAT_DICT
###################################################

# normalization by column ()

maximum = np.amax([i for v in feat_dict_headless_rescaled.values() for i in v], axis=0)
minimum = np.amin([i for v in feat_dict_headless_rescaled.values() for i in v], axis=0)
norm_feat_dict = {key: util.normalize(value, minimum, maximum) for key, value in feat_dict_headless_rescaled.items()}

# %%
###################################################
# SAVE FEAT_DICT
###################################################

with open('../data/feat_dict_ready.pkl', 'wb') as handle:
    pickle.dump(norm_feat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# %%
###################################################
# exps: rescale from 0.1s to 0.5s per label
'''SHIFTED TO pruning_deam.py'''
###################################################

# for rowidx in exps.index:
#     exps.at[rowidx, 'arousals'] = average_1D(np.array(exps.at[rowidx, 'arousals']), sample_factor)
#     exps.at[rowidx, 'valences'] = average_1D(np.array(exps.at[rowidx, 'valences']), sample_factor)


# %%
###################################################
# SAVE EXPS
###################################################
# exps.to_pickle(os.path.join(os.path.abspath('..'), 'data', 'exps_ready.pkl'))

# %%
'''
TEST
'''

feat_dict = util.load_pickle('../data/feat_dict_ready.pkl')

exps = pd.read_pickle('../data/exps_ready.pkl')

temp = exps.loc[0]
songurl = temp['songurl']
print('songurl: ', songurl)
print('feature shape: ', np.shape(feat_dict[songurl]))
print('a label shape: ', np.shape(temp['arousals']))
print('v label shape: ', np.shape(temp['valences']))

count = 0
for idx in exps.index:
    row = exps.loc[idx]
    count += len(row['arousals'])
print(count)

# %%
