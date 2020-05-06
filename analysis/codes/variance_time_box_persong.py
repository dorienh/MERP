#%%
#   imports
import os
import sys
# sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
print(sys.path)
import pickle
import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt


#%%
datatype = 'arousals'

#%%
# load data
exps = pd.read_pickle('../../data/exps_ready.pkl')

#load pinfo
pinfo = pd.read_pickle('../../data/pinfo.pkl')

# %%# %%

song_exps = exps[exps['songurl']=='deam_115']
print(len(exps['workerid'].unique())) # check number of participants and number of deam_115 

# %%
arousals = song_exps[datatype]
arousals = np.array([np.array(singlelist) for singlelist in arousals])

#%%
def extract_exp_by_song_to_numpy(exps, songurl, datatype):
    # get all rows of stipulated songurl
    song_exps = exps[exps['songurl']==songurl]
    num_participants = len(song_exps['workerid'].unique())
    print(f'number of participants labelled {songurl}: {num_participants}')
    # extract either arousals or valences and convert to numpy array to calc var
    labels = song_exps[datatype]
    labels = np.array([np.array(singlelist) for singlelist in labels])
    return labels, num_participants

# %%
'''
DEAM ONLY cos by timestep might be very messy for longer songs.
boxplot at each timestep for each song, of the labels across participants
'''
for songurl in util.songlist:   

    fig = plt.figure(figsize=(28,6))

    labels, _ = extract_exp_by_song_to_numpy(exps, songurl, datatype)

    plt.boxplot(labels)

    # plt.xticks(ticks=np.arange(1, 1+len(pindex_list)), labels=pindex_list)
    plt.ylabel(f'{datatype[:-1]}')
    plt.xlabel('time (0.5s)')
    plt.title(f'{datatype[:-1]} boxplots for song {songurl} at each time step over all participants')
    plt.savefig(f'../plots/deam_per_sec_boxplot_per_song/{datatype[:-1]}/{songurl}.png')
    plt.close()

# %%
'''
boxplot of variances at each timestep across participants
'''

#%%
def get_var_per_song(exps):
    variance_list = []
    for songurl in util.songlist:
        labels, _ = extract_exp_by_song_to_numpy(exps, songurl, datatype)
        variances = np.var(labels, axis=0)
        variance_list.append(variances)
    return variance_list

variance_list = get_var_per_song(exps)

# %%
# temp = variance_list[0]
fig = plt.figure(figsize=(28,6))
plt.boxplot(variance_list)

plt.xticks(ticks=np.arange( len(util.songlist)), labels=util.songlist)
plt.tick_params(axis='x', which='major', labelrotation=45)
plt.ylim(-0.5,0.5)

plt.ylabel(f'{datatype[:-1]}')
plt.xlabel('song')
plt.title(f'{datatype[:-1]} boxplots of variance-across-all-participants-at-each-timestep')
plt.tight_layout()
plt.savefig(f'../plots/variance_time_boxplot/{datatype[:-1]}.png')
plt.close()
# %%
