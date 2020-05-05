#%%
'''
imports
'''
import os
import glob
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

import util

#%%
feat_dict = util.load_pickle('../../data/feat_dict.pkl')

## 2) use feat_dict to find number of timesteps in each song, store in feat_len_dict
def count_timestep_feat_dict(feat_dict):
    feat_len_dict = {}
    for key, val in feat_dict.items():
        feat_len_dict[key] = len(val)
    return feat_len_dict

feat_len_dict = count_timestep_feat_dict(feat_dict)

## 3) load the amazon data
exps = pd.read_pickle(os.path.join(os.path.abspath('../..'), 'data', 'mediumrare', 'unpruned_exps.pkl'))
pinfo = pd.read_pickle(os.path.join(os.path.abspath('../..'), 'data', 'mediumrare', 'unpruned_pinfo.pkl'))

#%%
###################################################
# count number of features shorter than 2 seconds
###################################################

counter = 0
# for idx in exps.index:
#     featlen = feat_len_dict[exps.loc[idx,'songurl']]
#     if len(exps.loc[idx,'arousals']) <= featlen-20:
#         counter +=1
def too_short(exps, feat_len_dict, threshold=20):
    qualified_idexes = []

    for idx, exp in exps.iterrows():
        featlen = feat_len_dict[exp['songurl']] + 12
        
        if (len(exp['arousals']) <= featlen-threshold):
            qualified_idexes.append(idx)
    # print('num qualified entries: ', len(qualified_idexes))
    return exps.iloc[qualified_idexes].reset_index(drop=True)

tmpexps = too_short(exps,feat_len_dict)

print(counter)
print(len(exps))
count=0
for idx in exps.index:
    row = exps.loc[idx]
    count += len(row['arousals'])
print(count)

print(len(tmpexps))
count=0
for idx in tmpexps.index:
    row = tmpexps.loc[idx]
    count += len(row['arousals'])
print(count)
        


# %%

###################################################
# openSMILE feature length vs actual song length
###################################################
import scipy.io.wavfile as wav

####
# finding the actual song length in 0.1 seconds.
####
def gen_actual_songlen_dict():
    actual_songlen = {}
    songspath = "../../data/50songs"
    songdirs = os.listdir(songspath)
    for songdir in songdirs:
        subpath = os.path.join(songspath,songdir,'*.wav')
        for song in glob.glob(subpath):

            (source_rate, source_sig) = wav.read(song)
            duration_deciseconds = round(len(source_sig) / float(source_rate) *10, 2)

            songname = os.path.basename(song)[:-4]
            actual_songlen[f'{songdir}_{songname}'] = duration_deciseconds
    return actual_songlen

actual_songlen = gen_actual_songlen_dict()
####
#feat_len_dict vs actual_songlen...
####

len_discrepancy_dict = {key: round(actual_songlen[key] - feat_len_dict.get(key, 0),2) for key in actual_songlen}


# %%
