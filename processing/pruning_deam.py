'''
Different pruning methods. Based on statistics instead.
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
feat_dict = util.load_pickle('../data/feat_dict.pkl')

## 2) use feat_dict to find number of timesteps in each song, store in feat_len_dict
def count_timestep_feat_dict(feat_dict):
    feat_len_dict = {}
    for key, val in feat_dict.items():
        feat_len_dict[key] = len(val)
    return feat_len_dict

feat_len_dict = count_timestep_feat_dict(feat_dict)

## 3) load the amazon data
exps = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'mediumrare', 'unpruned_exps.pkl'))
pinfo = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'mediumrare', 'unpruned_pinfo.pkl'))


'''
1) remove trials that are too short or too long (remove by index)
'''
## 4)  keep trials that are >= feat_len but <= feat_len+20
def too_short_too_long(exps, feat_len_dict, threshold=20):
    qualified_idexes = []

    for idx, exp in exps.iterrows():
        featlen = feat_len_dict[exp['songurl']]
        
        if (len(exp['arousals']) >= featlen) and (len(exp['arousals']) <= featlen+threshold) :
            qualified_idexes.append(idx)
    # print('num qualified entries: ', len(qualified_idexes))
    return exps.iloc[qualified_idexes].reset_index(drop=True)

exps2 = too_short_too_long(exps, feat_len_dict)


'''
2) remove trials under erroneous profiles (remove by workerid and batch)
'''
def erroneous_training_duration_profiles(pinfo, exps):
    # check training duration
    # remove if more than 100 or less than 0
    pinfo_td = pinfo['training_duration'].astype(int)
    err_p = pinfo[(pinfo_td < 0) | (pinfo_td > 100)]

    for _, participant in err_p.iterrows():
        exps = exps[~ ((exps['workerid'] == participant['workerid']) & (exps['batch'] == f"{participant['batch']}"))]
    return exps.reset_index(drop=True)

exps3 = erroneous_training_duration_profiles(pinfo, exps2)

'''
# repeat workerid, different answers. remove. 
'''
def remove_duplicate_conflicting_profiles(pinfo, exps):
    duplicate_pinfos = pinfo[pinfo['workerid'].duplicated(keep=False)]
    duplicate_pinfos = duplicate_pinfos.drop(columns='batch')
    duplicate_wids = duplicate_pinfos['workerid'].unique()
    to_delete = []
    for wid in duplicate_wids:
        temp = duplicate_pinfos[duplicate_pinfos['workerid']==wid]
        # print(temp.iloc[0].values[1::])
        # print(temp.iloc[1].values[1::])
        # print()
        if not np.array_equal(temp.iloc[0].values[1::],temp.iloc[1].values[1::]):
            to_delete.append(wid)
        
    for wid in to_delete:
        exps = exps[~ (exps['workerid'] == wid)]
    
    return exps.reset_index(drop=True)

exps3 = remove_duplicate_conflicting_profiles(pinfo, exps3)
    
'''
# training 'No' but training_duration > 0 (3 workerids)
'''
def remove_training_conflicting_profiles(pinfo, exps):
    err_p = pinfo.loc[(pinfo['training_duration'].astype(int)>0) & (pinfo['training']=='No')]
    to_delete = err_p['workerid'].tolist()

    for wid in to_delete:
        exps = exps[~ (exps['workerid'] == wid)]
    
    return exps.reset_index(drop=True)

exps3 = remove_training_conflicting_profiles(pinfo, exps3)

def remove_missing_deam_workers(exps):
    to_delete = []

    for workerid, group in exps.groupby('workerid'):
        songlist = group['songurl'].unique()
        if not all(song in songlist for song in ['deam_115', 'deam_343', 'deam_745', 'deam_1334']):
            to_delete.append(workerid)
    
    for wid in to_delete:
        exps = exps[~ (exps['workerid'] == wid)]
    
    return exps.reset_index(drop=True)

exps3= remove_missing_deam_workers(exps3)

print(f'number of trials: {len(exps3)}')
print(f'number of workers: {len(exps3["workerid"].unique())}')


#%%
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

'''
get deam trials from exps
'''
deamtrials = exps3[exps3['songurl'].str.contains('deam')].reset_index()

def average_1D(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def rescale_and_remove_head_deam(trial, desired_len=60):
    # rescale
    arousal = average_1D(trial['arousals'], 5)
    valence = average_1D(trial['valences'], 5)
    # remove head
    lendiff = len(arousal) - desired_len
    arousal = arousal[lendiff::]
    valence = valence[lendiff::]

    return arousal, valence

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


#%%
'''
DEAM COMPARISON
define functions to check which workerids comply with arbitruary tresholds for each statistic type.
'''
# STD
def worker_comply_deam_std(deamtrials, std_mult, std_threshold, song_threshold):
    qualified_workerids = []
    for workerid, wid_group in deamtrials.groupby('workerid'):
        
        # dict to store booleans if the worker complies for the 4 songs. default is False, 0
        complaince_dict = {'deam_115':{'a':0, 'v':0}, 'deam_343':{'a':0, 'v':0}, 'deam_745':{'a':0, 'v':0}, 'deam_1334':{'a':0, 'v':0}}

        for idx in wid_group.index:
            trial = wid_group.loc[idx]
            songurl = trial['songurl']
            temp_a,temp_v = rescale_and_remove_head_deam(trial)

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
#%%
from scipy.stats import pearsonr
def worker_comply_deam_pearson(deamtrials, song_threshold):
    qualified_workerids = []
    for workerid, wid_group in deamtrials.groupby('workerid'):
            
        # dict to store booleans if the worker complies for the 4 songs. default is False, 0
        complaince_dict = {'deam_115':{'a':0, 'v':0}, 'deam_343':{'a':0, 'v':0}, 'deam_745':{'a':0, 'v':0}, 'deam_1334':{'a':0, 'v':0}}

        for idx in wid_group.index:
            trial = wid_group.loc[idx]
            songurl = trial['songurl']

            temp_a,temp_v = rescale_and_remove_head_deam(trial)
            r_a, _ = pearsonr(deamstats[datatypes[0]][songurl]['ave'], temp_a)
            r_v, _ = pearsonr(deamstats[datatypes[1]][songurl]['ave'], temp_v)

            if not np.isnan(r_a):
                complaince_dict[trial['songurl']]['a'] = 1
            if not np.isnan(r_v):
                complaince_dict[trial['songurl']]['v'] = 1
            
    # check if both arousal and valence are within the threshold
        complaince_dict_combined = {}
        for key, val in complaince_dict.items():
            # if sum == 1 meaning both are 1. haha
            if sum(val.values()) == 2:
                complaince_dict_combined[key] = 1
            else:
                complaince_dict_combined[key] = 0
        

        if sum(complaince_dict_combined.values()) > song_threshold:
                qualified_workerids.append(deamtrials.loc[idx, 'workerid'])
    return qualified_workerids

pearson_qualified_workers = worker_comply_deam_pearson(deamtrials, song_threshold = 2)
print(len(pearson_qualified_workers))
#%%
'''
define a function that accepts lists of qualified workers, 
return a list of workers that are found in every list.  
'''
def common_qualified_workers(*qualified_worker_lists):
    result = set(qualified_worker_lists[0])
    for s in qualified_worker_lists[1:]:
        result.intersection_update(s)
    return result

qualified_workers = common_qualified_workers(std_qualified_workers, pearson_qualified_workers)

print('number of qualified workers: ', len(qualified_workers))


#%%
'''
get qualified trials from exps according to qualified_workers list.
'''

qualified_trials = exps3[exps3['workerid'].isin(qualified_workers)].reset_index()

print(f'number of trials: {len(exps3)}')
print(f'number of workers: {len(exps3["workerid"].unique())}')
print(f'number of trials: {len(qualified_trials)}')
print(f'number of workers: {len(qualified_trials["workerid"].unique())}')

#%%

'''
rescale and remove head
'''
## 1) load feat_dict
feat_dict_ready = util.load_pickle('../data/feat_dict_ready.pkl')

## 2) use feat_dict to find number of timesteps in each song, store in feat_len_dict
feat_len_dict_ready = count_timestep_feat_dict(feat_dict_ready)

def rescale_and_remove_head(trial, song_len_dict):
    songurl = trial['songurl']
    song_len = song_len_dict[songurl]

    # rescale
    arousal = average_1D(trial['arousals'], 5)
    valence = average_1D(trial['valences'], 5)
    # remove head
    startidx = len(arousal) - song_len
    arousal = arousal[startidx::]
    valence = valence[startidx::]
    return arousal, valence

def rescale_and_remove_head_dataframe(exps, song_len_dict):
    exps = exps.copy()
    for idx in exps.index:
        trial = exps.loc[idx]
        arousal, valence = rescale_and_remove_head(trial, feat_len_dict_ready)
        exps.at[idx,'arousals'] = arousal
        exps.at[idx,'valences'] = valence
    return exps

modified_exps = rescale_and_remove_head_dataframe(qualified_trials, feat_len_dict_ready)

#%%
def check_exps(exps):
    count = 0
    for idx in exps.index:
        row = exps.loc[idx]
        count += len(row['arousals'])
    print(f'total number of timesteps: {count}')
    print(f'number of trials: {len(exps)}')
    print(f'number of workers: {len(exps["workerid"].unique())}')

print('after removing erronous profiles')
check_exps(exps3)
print('\nafter removing based on std and pearson')
check_exps(qualified_trials)
print('\nafter rescaling and removing first 15 seconds')
check_exps(modified_exps)


#%%
###################################################
# NORMALIZE EXPS (they are already normalized naturally so no need.)
###################################################
# temp = modified_exps['arousals'].to_numpy()
# np.max(np.concatenate(temp))

# %%
import pickle
with open('../data/exps_ready.pkl', 'wb') as handle:
    pickle.dump(modified_exps, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%