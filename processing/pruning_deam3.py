'''
Using the other config file, hence features were alr in 0.5s but exps still needs rescaling.
'''

#%%
'''
imports
'''
import os
import glob
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
import util

'''
loading files
'''
## 1) load feat_dict
feat_dict = util.load_pickle('../data/feat_dict2.pkl')
feat_dict_ready = util.load_pickle('../data/feat_dict_ready2.pkl')

#%%
## 2) use feat_dict to find number of timesteps in each song, store in feat_len_dict
def count_timestep_feat_dict(feat_dict):
    feat_len_dict = {}
    for key, val in feat_dict.items():
        feat_len_dict[key] = len(val)
    return feat_len_dict

feat_len_dict = count_timestep_feat_dict(feat_dict)
feat_ready_len_dict = count_timestep_feat_dict(feat_dict_ready)

## 3) load the amazon data
exps = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'mediumrare', 'unpruned_exps.pkl'))
pinfo = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'mediumrare', 'semipruned_pinfo.pkl'))

def printcounts(exps):
    nonmaster = exps[(exps['batch'] == '7') | (exps['batch'] == '8') ]
    master = exps[(exps['batch'] == '4') | (exps['batch'] == '5') | (exps['batch'] == '6')]

    print('master:')
    print('num wid: ', len(master['workerid'].unique()))
    print('num trial: ', len(master))
    print('non master:')
    temp = len(nonmaster['workerid'].unique()) - 1
    print('num wid: ', temp)
    print('num trial: ', len(nonmaster))
    print('total')
    print('num wid: ', len(exps['workerid'].unique()))
    print('num trials: ', len(exps))

# print('mean songs labelled per wid: ', exps.groupby('workerid').count().mean())

# %%
'''
1) only keep entries with workerids in semipruned_pinfo
erroneous profiles have been removed.
'''
exps2 = exps[exps['workerid'].isin(pinfo.workerid)].reset_index(drop=True)



# %%
'''
rescale to 0.5s timesteps
'''
def average_1D(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def rescale(trial):
    songurl = trial['songurl']
    # rescale
    arousal = average_1D(trial['arousals'], 5)
    valence = average_1D(trial['valences'], 5)

    return arousal, valence

def rescale_dataframe(exps):
    exps = exps.copy()
    for idx in exps.index:
        trial = exps.loc[idx]
        arousal, valence = rescale(trial)
        exps.at[idx,'arousals'] = arousal
        exps.at[idx,'valences'] = valence
    return exps

exps3 = rescale_dataframe(exps2)

# %%
## 4)  keep trials that are >= feat_len but <= feat_len+ 2 seconds
def too_short_too_long(exps, feat_len_dict, threshold=4):
    qualified_idexes = []

    for idx, exp in exps.iterrows():
        featlen = feat_len_dict[exp['songurl']]
        
        if (len(exp['arousals']) >= featlen) and (len(exp['arousals']) <= featlen+threshold) :
            qualified_idexes.append(idx)
    # print('num qualified entries: ', len(qualified_idexes))
    return exps.iloc[qualified_idexes].reset_index(drop=True)

exps4 = too_short_too_long(exps3, feat_len_dict)

# %%
def remove_missing_deam_workers(exps):
    to_delete = []

    for workerid, group in exps.groupby('workerid'):
        songlist = group['songurl'].unique()
        if not all(song in songlist for song in ['deam_115', 'deam_343', 'deam_745', 'deam_1334']):
            to_delete.append(workerid)
    
    for wid in to_delete:
        exps = exps[~ (exps['workerid'] == wid)]
    
    return exps.reset_index(drop=True)

exps5 = remove_missing_deam_workers(exps4)

# %%
'''
remove head
'''

def remove_head(trial, song_len):
    # remove head
    arousal = trial['arousals']
    valence = trial['valences']

    startidx = len(arousal) - song_len
    arousal = arousal[startidx::]
    valence = valence[startidx::]
    return arousal, valence

def dehead_dataframe(exps, song_len_dict):
    exps = exps.copy()
    for idx in exps.index:
        trial = exps.loc[idx]
        songurl = trial['songurl']
        song_len = song_len_dict[songurl]

        arousal, valence = remove_head(trial, song_len)
        exps.at[idx,'arousals'] = arousal
        exps.at[idx,'valences'] = valence
    return exps

exps6 = dehead_dataframe(exps5, feat_ready_len_dict)

# %%
'''
get deam trials from exps
'''
deamtrials = exps6[exps6['songurl'].str.contains('deam')].reset_index()

# %%
'''
DEAM COMPARISON
retreive DEAM data.
'''
datatypes = ['arousals', 'valences']
deampath = '../data/deam_annotations/annotations_per_each_rater/dynamic-persec'
deamsonglist = ['115', '343', '745', '1334']

def get_deam_annotations(datatypes, deampath, deamsonglist):
    deamlabels = {datatypes[0]:{}, datatypes[1]:{}}
    for datatype in datatypes:
        for deamsong in deamsonglist:
            path = os.path.join(deampath, datatype[:-1], f'{deamsong}.csv')
            print(path)
            deamlabels[datatype][f'deam_{deamsong}'] =  pd.read_csv(path, index_col=None, header=0) 
    return deamlabels

def get_deamstats(datatype, deamlabels):
    deamstats = {datatypes[0]:{}, datatypes[1]:{}}
    for datatype in deamlabels.keys():
        for songurl, df in deamlabels[datatype].items():
            ave = df.mean(axis=0)
            stddev = df.std(axis=0)
            deamstats[datatype][songurl] = {'ave':ave.to_numpy(), 'stddev':stddev.to_numpy()}
    return deamstats

deamlabels = get_deam_annotations(datatypes, deampath, deamsonglist)
deamstats = get_deamstats(datatypes, deamlabels)

#%%
'''
DEAM COMPARISON
define functions to check which workerids comply with arbitruary tresholds for each statistic type.
'''
# STD

def check_if_within_std(deamstat, ourlabel, std_mult=1):
    maxlist = deamstat['ave'] + deamstat['stddev']*std_mult
    minlist = deamstat['ave'] - deamstat['stddev']*std_mult

    within_bool_list = []

    for mindeam, ours, maxdeam in zip(minlist, ourlabel, maxlist):
        if mindeam < ours < maxdeam:
            within_bool_list.append(True)
        else:
            within_bool_list.append(False)
    return within_bool_list


def worker_comply_deam_std(deamtrials, std_mult, std_threshold, song_threshold):
    qualified_workerids = []
    for workerid, wid_group in deamtrials.groupby('workerid'):
        
        # dict to store booleans if the worker complies for the 4 songs. default is False, 0
        complaince_dict = {'deam_115':{'a':0, 'v':0}, 'deam_343':{'a':0, 'v':0}, 'deam_745':{'a':0, 'v':0}, 'deam_1334':{'a':0, 'v':0}}

        for idx in wid_group.index:
            trial = wid_group.loc[idx]
            songurl = trial['songurl']
            temp_a = trial['arousals']
            temp_v = trial['valences']

            check_a = check_if_within_std(deamstats[datatypes[0]][songurl], temp_a, std_mult=std_mult)
            check_v = check_if_within_std(deamstats[datatypes[1]][songurl], temp_v, std_mult=std_mult)
            percentage_match_a = round(sum(check_a)/len(check_a)*100, 2)
            percentage_match_v = round(sum(check_v)/len(check_v)*100, 2)

            if percentage_match_a >= std_threshold:
                complaince_dict[trial['songurl']]['a'] = 1
            if percentage_match_v >= std_threshold:
                complaince_dict[trial['songurl']]['v'] = 1
        
        # check if both arousal and valence are within the threshold
        complaince_dict_combined = {}
        for key, val in complaince_dict.items():
            # sum >=1 meaning, at least arousal or valence was within the std.
            if sum(val.values()) >= 1:
                complaince_dict_combined[key] = 1
            else:
                complaince_dict_combined[key] = 0
            
        if sum(complaince_dict_combined.values()) == song_threshold:
            qualified_workerids.append(deamtrials.loc[idx, 'workerid'])
    return qualified_workerids

std_qualified_workers = worker_comply_deam_std(deamtrials, std_mult=2, std_threshold=50.0, song_threshold=4)
print(len(std_qualified_workers))


# %%
qualified_trials = exps6[exps6['workerid'].isin(std_qualified_workers)].reset_index()

print('original')
print(f'number of trials: {len(exps)}')
print(f'number of workers: {len(exps["workerid"].unique())}')
print('after removing too short too long')
print(f'number of trials: {len(exps6)}')
print(f'number of workers: {len(exps6["workerid"].unique())}')
print('after removing deam std uncompliant')
print(f'number of trials: {len(qualified_trials)}')
print(f'number of workers: {len(qualified_trials["workerid"].unique())}')
# %%
util.save_pickle('../data/exps_ready3.pkl', qualified_trials)
# %%

# %%

# %%
