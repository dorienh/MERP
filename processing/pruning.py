
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

import util
#%%
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

#%%
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

#%%
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

#%%
# repeat workerid, different answers. remove. 
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
    
#%%
# training 'No' but training_duration > 0 (3 workerids)
def remove_training_conflicting_profiles(pinfo, exps):
    err_p = pinfo.loc[(pinfo['training_duration'].astype(int)>0) & (pinfo['training']=='No')]
    to_delete = err_p['workerid'].tolist()

    for wid in to_delete:
        exps = exps[~ (exps['workerid'] == wid)]
    
    return exps.reset_index(drop=True)

exps3 = remove_training_conflicting_profiles(pinfo, exps3)

#%%
'''
3) remove trials that have long plateaus (remove by index)
'''
def check_for_plateau(temp, timestep_threshold=1200):
    
    if all(e == temp[0] for e in temp):
        return True # stagnant throughout.

    assert type(timestep_threshold) == int, 'timestep_threshold entered must be an integer!'
    for i in range(len(temp)-timestep_threshold):
        curr = temp[i]
        if all(e == curr for e in temp[i:i+timestep_threshold]) == True:
            return True
    return False

def remove_stagnant(exps, timestep_threshold=1200):
    qualified_indexes = []
    for idx, exp in exps.iterrows():
        if (not check_for_plateau(exp['arousals'], timestep_threshold)) or (not check_for_plateau(exp['valences'], timestep_threshold)):
            qualified_indexes.append(idx)
    # print(len(qualified_indexes))
    return exps.iloc[qualified_indexes].reset_index(drop=True)

exps4 = remove_stagnant(exps3)


#%%
'''
4) remove trials under profiles that do not have all deam songs. (remove by workerid_batch)
'''

def deam_incomplete(exps):
    
    # create new column {"pid": workerid_batch} 
    exps["pid"] = exps["workerid"] + '_' + exps["batch"].map(str)
    # groupby "pid", count num deam_songs, if == 4 qualified.
    disqualified_pids = []
    for (pid, group) in exps.groupby("pid"):
        deam_trials = group[group['songurl'].str.contains('deam')]
        if len(deam_trials) != 4:
            disqualified_pids.append(pid)
    # remove disqualified pid trials from exps
    for pid in disqualified_pids:
        exps = exps[~ (exps['pid'] == pid)]
    # drop pid column
    exps = exps.drop(columns=['pid'])

    return exps.reset_index(drop=True)


exps5 = deam_incomplete(exps4)


#%%
'''
5) participant's valence/arousal range is less that 1/3 of possible range, remove.
by this stage of pruning, arousal ranges are all acceptible. 
5 workerids have small valence range. 
['A15XZ36IPGYY2N', 'A2B6OA4V8YABCZ', 'A2DXYZSG2E7HT0', 'A2TC1ZFCTOCDA5', 'A37V7156T1VXI7']
'''
def range_check(exps):
    disqualified_workerids = []
    for workerid, group in exps.groupby('workerid'):
        garousal = np.array(group['arousals'])
        # flatten
        garousal = np.concatenate(garousal, axis=None)
        # find difference between min and max
        # print(f"max: {min(garousal)} || min: {max(garousal)}")
        if (max(garousal) - min(garousal)) < (2/3):
            disqualified_workerids.append(workerid)
        # repeat for valence
        gvalence = np.array(group['valences'])
        gvalence = np.concatenate(gvalence, axis=None)
        if (max(gvalence) - min(gvalence)) < (2/3):
            disqualified_workerids.append(workerid)
    # print(disqualified_workerids)

    # remove disqualified workerid trials from exps
    for workerid in disqualified_workerids:
        exps = exps[~ (exps['workerid'] == workerid)]
    
    return exps.reset_index(drop=True)

exps6 = range_check(exps5)


#%%
'''
6) unify arousal and valence raw input lengths
'''
def uniformify_av_lengths(exps, feat_len_dict, cut_position='front'):
    
    for songid, group in exps.groupby('songurl'):
        featlen = feat_len_dict[songid]
        for rowidx, row in group.iterrows():
            # IF: cut off excess at the BACK
            if cut_position is 'back':
                exps.at[rowidx, 'arousals'] = row['arousals'][:featlen]
                exps.at[rowidx, 'valences'] = row['valences'][:featlen]
            # IF: cut off excess at the FRONT
            elif cut_position is 'front':
                start = len(row['arousals']) - featlen
                exps.at[rowidx, 'arousals'] = row['arousals'][start:]
                exps.at[rowidx, 'valences'] = row['valences'][start:]
            # IF: cut off excess at BOTH front and back
            elif cut_position is 'both':
                start = (len(row['arousals']) - featlen)//2
                end = start + featlen
                exps.at[rowidx, 'arousals'] = row['arousals'][start:end]
                exps.at[rowidx, 'valences'] = row['valences'][start:end]
            else:
                print('invalid cut_position parameter given.')
        
    return exps

exps7 = uniformify_av_lengths(exps6, feat_len_dict)


#%%
'''
7) handle deam duplicates, remove the duplicate with lower correlation.
calc the corr values for all 4 songs, sum them per batch, remove the one with lower corr values.
'''
def trials_wif_duplicated_songs(exps):
    dup_idx = []
    for wid, group in exps.groupby('workerid'):
        songurls = group['songurl']
        bool_list = songurls.duplicated(keep=False)
        if any(bool_list):
            dup_idx.append(songurls[bool_list].index.tolist())
    # flatten list of index lists to one list.
    dup_idx = [item for sublist in dup_idx for item in sublist]
    dup_exps = exps.iloc[dup_idx]
    return dup_exps

def get_deam_arousal_means():
    '''
    Extract the average Arousal and Valence arrays of DEAM songs from official DEAM dataset.
    '''
    def csv2df(filepath):
        df = pd.read_csv(filepath)
        # print('reading file!! shape of resulting df: ', df.shape)
        return df

    deam_arousals = csv2df(filepath="../data/deam_annotations/annotations_averaged_per_song/dynamic_per_second_annotations/arousal.csv")
    # /home/meowyan/Documents/emotion/data/deam_annotations/annotations_averaged_per_song/dynamic_per_second_annotations/arousal.csv
    deamsongids = [115.0, 343.0, 745.0, 1334.0]
    deam_ave_ars = {}
    for sid in deamsongids:
        songid = 'deam_{}'.format(str(int(sid)))
        temp_ars = deam_arousals[deam_arousals['song_id'] == sid].to_numpy().transpose()[1::]
        deam_ave_ars[songid] = temp_ars[~np.isnan(temp_ars)]
        # print('songid {}, min arousal: {}, max arousal: {}, min valence: {}, max valence: {}'.format(songid,min(temp_ars),max(temp_ars),min(temp_vl),max(temp_vl)))
    return deam_ave_ars

def remove_duplicate_trials(exps):
    deam_ave_ars = get_deam_arousal_means()

    from scipy.stats import pearsonr

    dup_exps = trials_wif_duplicated_songs(exps)

    deam_exps = dup_exps[dup_exps['songurl'].str.contains('deam')]

    disqualified_list = [] # [(workerid,batch)]
    for wid, wgroup in deam_exps.groupby('workerid'):

        # initialize correlation sums as 0
        corr_dict = dict.fromkeys(wgroup['batch'].tolist(), 0) 

        for songurl, sgroup in wgroup.groupby('songurl'):
            # print(f"workerid: {wid} || songurl: {songurl}")
            
            for index, row in sgroup.iterrows():
                # deam labels are per 0.5 while our labels are per 0.1 so resample ours
                resampled_ars = list(np.mean(row['arousals'][138::].reshape(-1, 5), 1)) #138 because 15 seconds were cut off in the deam dataset.
                corr_coeff, p_value = pearsonr(resampled_ars, deam_ave_ars[songurl])
                if np.isnan(corr_coeff):
                    corr_coeff = 0
                corr_dict[row['batch']] += abs(corr_coeff)
            # print(corr_dict)
        disqualified_batch = max(corr_dict, key=corr_dict.get)
        disqualified_list.append((wid, disqualified_batch))
    # print(disqualified_list)

    for workerid, batch in disqualified_list:
        dup_exps = dup_exps[~ ((dup_exps['workerid'] == workerid) & (dup_exps['batch'] == batch)) ]
    # print(disqualified_exps)
        # for index, row in dup_exps.iterrows():
            # print(index, row)
    return exps[~exps.index.isin(dup_exps.index)]


exps8 = remove_duplicate_trials(exps7)

#%%
'''
8) export to hdf...
'''
exps8.to_pickle(os.path.join(os.path.abspath('..'), 'data', 'exps.pkl'))

#%%
'''
9) rescale by person and export to hdf
'''
def normalize(data, minimum, maximum):
    ''' function to normalize the values of an array to [-1, 1] given min and max'''
    data = np.array(data)
    return ((data - minimum)/(maximum - minimum))*2 - 1

def find_minmax_per_group(group, datatype):

    values = group[datatype].ravel()
    values = np.concatenate(values)
    minval = min(values)
    maxval = max(values)

    return minval, maxval

def rescale_by_workerid(exps):
    exps_copy = exps.copy(deep=True)
    a_scaled_dict = {} # {index:[arousal]}
    v_scaled_dict = {} # {index:[valence]}
    for workerid, group in exps_copy.groupby('workerid'):
        # print(workerid)
        a_minval, a_maxval = find_minmax_per_group(group, 'arousals')
        v_minval, v_maxval = find_minmax_per_group(group, 'valences')
        # print(minval, maxval)
        for index, row in group.iterrows():
            a_rescaled = normalize(row['arousals'], a_minval, a_maxval)
            v_rescaled = normalize(row['valences'], v_minval, v_maxval)
            # print(f"old: {row['arousals'][0]} || new: {rescaled_ars[0]}")
            a_scaled_dict[index] = a_rescaled
            v_scaled_dict[index] = v_rescaled
    for ind in exps_copy.index:
        exps_copy.at[ind, 'arousals'] = a_scaled_dict[ind]
        exps_copy.at[ind, 'valences'] = v_scaled_dict[ind]
    # scaled_df = pd.DataFrame.from_dict(scaled_dict, orient='index')
    # print(scaled_df)
    # exps_scaled = exps.assign(arousals=scaled_df[0])

    return exps_copy

    # i need to repeat for valence later. 

def check_minmax_per_workerid(exps):
    for workerid, group in exps.groupby('workerid'):
        minval, maxval = find_minmax_per_group(group, 'arousals')
        print(workerid, minval, maxval)
        
exps9 = rescale_by_workerid(exps8)

#%%
check_minmax_per_workerid(exps9)

#%%     
'''
export scaled data to h5 file
'''
exps9.to_pickle(os.path.join(os.path.abspath('..'), 'data', 'exps_rescaled.pkl'))


#%%

'''
10) smooth the input
'''
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
import pandas as pd

def smoothen(labelarray):
    smoothlabels = savgol_filter(labelarray, 15, 2, mode='nearest')
    smoothlabels = [1 if e > 1 else -1 if e < -1 else e for e in smoothlabels]
    return smoothlabels

exps_path = os.path.join(os.path.abspath('..'), 'data', 'exps_rescaled.pkl')
exps_scaled = pd.read_pickle(exps_path)
a_smoothed_dict = {} # {index:[arousal]}
v_smoothed_dict = {} # {index:[valence]}
for ind in exps_scaled.index:
    a_smooth = smoothen(exps_scaled.at[ind, 'arousals'])
    v_smooth = smoothen(exps_scaled.at[ind, 'valences'])
    a_smoothed_dict[ind] = a_smooth
    v_smoothed_dict[ind] = v_smooth
for ind in exps_scaled.index:
    exps_scaled.at[ind, 'arousals'] = a_smoothed_dict[ind]
    exps_scaled.at[ind, 'valences'] = v_smoothed_dict[ind]

exps_scaled.to_pickle(os.path.join(os.path.abspath('..'), 'data', 'exps_rescaled_smoothed.pkl'))

# tempidx = 7
# temp = smoothen(exps_scaled.at[tempidx, 'arousals'])
# plt.plot(exps_scaled.at[tempidx, 'arousals'], label='original')
# plt.plot(temp, label='smoothed')
# plt.legend()
# plt.show()


#%%
"""
1) remove trials that are too short or too long (remove by index)
2) remove trials under erroneous profiles (remove by workerid)
3) remove trials that have long plateaus (remove by index)
4) remove trials under profiles that do not have all deam songs. (remove by workerid_batch)
5) participant's valence/arousal range is less that 1/3 of possible range, remove.
6) unify arousal and valence raw input lengths
7) handle deam duplicates, remove the duplicate with lower correlation.
8) export to hdf...
9) rescale per workerid
10) smooth the input
"""

#%%
def printcounts(exps):
    nonmaster = exps[(exps['batch'] == '7') | (exps['batch'] == '8') ]
    master = exps[(exps['batch'] == '4') | (exps['batch'] == '5') | (exps['batch'] == '6')]

    print('unique wid count:')
    print('master: ', len(master['workerid'].unique()))
    print('nonmaster: ', len(nonmaster['workerid'].unique()))
    print('trial count:')
    print('master: ', len(master))
    print('nonmaster: ', len(nonmaster))
    print('total')
    print('num wid: ', len(exps['workerid'].unique()))
    print('num trials: ', len(exps))




# %%
count = 0
for idx in exps.index:
    row = exps.loc[idx]
    count += len(row['arousals'])
print(count)


# %%
'''
prune pinfo according to pruned exps!
'''
import util
import numpy as np
import pandas as pd
exps = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'exps.pkl'))
pinfo = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'mediumrare', 'unpruned_pinfo.pkl'))
pinfo = pinfo.drop(columns='batch')

#%%
workerids = pd.Series(exps['workerid'].unique(), name='workerid')

# %%

pruned_pinfo = pd.merge(pinfo, workerids, how='right', on='workerid')
pruned_pinfo2 = pruned_pinfo.drop_duplicates().reset_index(drop=True)


# %%
pruned_pinfo2.to_pickle(os.path.join(os.path.abspath('..'), 'data', 'pinfo.pkl'))


# %%
