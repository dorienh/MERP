
'''
Valence per song per time step

'''

#%%
#   imports
import os
import sys
sys.path.append(os.path.abspath('..'))
print(sys.path)
import pickle
import numpy as np
import pandas as pd
import util
import matplotlib.pyplot as plt


#%%
datatype = 'valences'

#%%
# load data
exps = pd.read_pickle('exps_ready.pkl')

#load pinfo
pinfo = pd.read_pickle('pinfo.pkl')

# %%

song_exps = exps[exps['songurl']=='10_130']
print(len(exps['workerid'].unique())) # check number of participants and number of deam_115 

# %%
arousals = song_exps[datatype]
arousals = np.array([np.array(singlelist) for singlelist in arousals])
# %%
temp = np.var(arousals, axis=0)

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

def variance_one_song(labels_matrix):
    # axis=0 across participants rather than time.
    variance = np.var(labels_matrix, axis=0)
    return variance

def simple_variance_plot(var_toplot, songurl, datatype, num_participants):
    plt.plot(var_toplot)
    plt.title(f'Variance of {datatype[:-1]} for {songurl} across {num_participants} participants')
    plt.ylim(0,0.5)
    plt.xlabel('time')
    plt.ylabel(f'variance')
    return plt

# simple_variance_plot(temp, '10_130', datatype, 64)

#%%
'''
time plots
2) each song, per time stamp, variance (0,0.5 range (probably)) x: time, y: valence variance/arousal variance 
'''
for songurl in util.songlist:
    labels_matrix, num_participants = extract_exp_by_song_to_numpy(exps, songurl, datatype)
    variance = variance_one_song(labels_matrix)
    plt = simple_variance_plot(variance, songurl, datatype, num_participants)
    # plt.show()
    plt.savefig(f'../analysis_plots/variance_per_song/{datatype[:-1]}/{songurl}.png')
    plt.close()

#%%
'''
time plots
1) each song, per time, average value (-1,1 range) x: time, y: valence/arousal (+- standard deviation)
'''
for songurl in util.songlist:
    labels_matrix, num_participants = extract_exp_by_song_to_numpy(exps, songurl, datatype)
    average = np.average(labels_matrix, axis=0)
    # std = sqrt(mean(abs(x - x.mean())**2))
    stddev = np.std(labels_matrix, axis=0)
    plt.plot(average, label='ave')
    plt.plot(average+stddev, label='ave+std')
    plt.plot(average-stddev, label='ave-std')
    plt.ylim(-1.1,1.1)
    plt.legend()
    plt.xlabel('time')
    plt.ylabel(f'{datatype[:-1]}')
    plt.title(f'Average of {datatype} for {songurl} across {num_participants} participants')
    # plt.show()
    plt.savefig(f'../analysis_plots/average_stddev_per_song/{datatype[:-1]}/{songurl}.png')
    plt.close()



#%%
def get_pindex(pinfo, workerid):
    indexlist = pinfo.index[pinfo['workerid'] == workerid].tolist()
    return indexlist[0]

# test function get_pindex
get_pindex(pinfo, 'A14W0AXTJ3R19V')

#%%
def extract_exp_by_song_w_participant(exps, songurl, datatype):

    # get all rows of stipulated songurl
    song_exps = exps[exps['songurl']==songurl]
    # get the number of participants that labelled the song
    num_participants = len(song_exps['workerid'].unique())
    # print(f'number of participants labelled {songurl}: {num_participants}')

    # extract either arousals or valences and convert to numpy array to calc var
    labels = song_exps[datatype]
    labels = [np.array(singlelist) for singlelist in labels]

    participant_list = song_exps['workerid'].unique()

    return labels, participant_list


def get_yticks_pindex_list(pinfo, participant_list):
    pindex_list = []
    for participant in participant_list:
        pidx = get_pindex(pinfo, participant)
        # pindex_list.append(f'p{pidx}')
        pindex_list.append(pidx)
    return pindex_list


#%%
'''
boxplots
1) for each song (non deam), average label of each participant 
'''
for songurl in util.songlist:
    labels, participant_list = extract_exp_by_song_w_participant(exps, songurl, datatype)
    pindex_list = get_yticks_pindex_list(pinfo, participant_list)


    fig = plt.figure(figsize=(28,6))

    plt.boxplot(labels)

    plt.xticks(ticks=np.arange(1, 1+len(pindex_list)), labels=pindex_list)
    plt.ylim(-1.1,1.1)
    plt.ylabel(f'{datatype[:-1]}')
    plt.xlabel('participant index')
    plt.title(f'{datatype[:-1]} boxplots for song {songurl} of each participant')
    # plt.show()
    plt.savefig(f'../analysis_plots/boxplot_label_per_song/{datatype[:-1]}/{songurl}.png')
    plt.close()

#%%
'''
time variance plots
1B) for each song (non deam), average variance of each participant 
'''
for songurl in util.songlist:
    labels, participant_list = extract_exp_by_song_w_participant(exps, songurl, datatype)
    
    variances_per_participant = np.var(labels, axis=1)

    pindex_list = get_yticks_pindex_list(pinfo, participant_list)

    fig = plt.figure(figsize=(28,6))

    plt.plot(np.arange(len(variances_per_participant)), variances_per_participant, 'o')

    plt.xticks(ticks=np.arange(0, len(pindex_list)), labels=pindex_list)
    plt.ylim(-0.02,0.8)
    plt.ylabel(f'variance of {datatype[:-1]}')
    plt.xlabel('participant index')
    plt.title(f'{datatype[:-1]} variances of each participant for song {songurl}')
    # plt.show()
    
    plt.savefig(f'../analysis_plots/variance_per_p_per_song/{datatype[:-1]}/{songurl}.png')
    plt.close()
    

#%%
'''
boxplots
2) for each participant, average label of each song
'''
workerid_list = exps['workerid'].unique()
for workerid in workerid_list:
    wid_exps = exps[exps['workerid']==workerid]

    labels = wid_exps[datatype]
    labels = [np.array(singlelist) for singlelist in labels]

    song_list = wid_exps['songurl'].tolist()

    fig = plt.figure(figsize=(12,6))
    plt.boxplot(labels)
    plt.xticks(ticks=np.arange(1, 1+len(song_list)), labels=song_list)
    plt.ylim(-1.1,1.1)
    plt.ylabel(f'{datatype[:-1]}')
    plt.xlabel('song')
    plt.title(f'{datatype[:-1]} boxplots for participant {workerid} of each song')
    # plt.show()
    plt.savefig(f'../analysis_plots/boxplot_label_per_participant/{datatype[:-1]}/{workerid}.png')
    plt.close()


#%%
'''
time variance plots
2B) for each participant, average variance of each song
'''
workerid_list = exps['workerid'].unique()

for workerid in workerid_list:
    wid_exps = exps[exps['workerid']==workerid]
    print(workerid)

    labels = wid_exps[datatype]
    labels = [np.array(singlelist) for singlelist in labels]
    variances_per_song = [np.var(label) for label in labels]

    song_list = wid_exps['songurl'].tolist()

    fig = plt.figure(figsize=(12,6))
    plt.plot(variances_per_song, 'o')
    plt.xticks(ticks=np.arange( len(song_list)), labels=song_list)
    plt.ylim(-0.02, 0.8)
    plt.ylabel(f'variance of {datatype[:-1]}')
    plt.xlabel('song')
    plt.title(f'{datatype[:-1]} variance of each song by participant {workerid}')
    # plt.show()
    
    plt.savefig(f'../analysis_plots/variance_per_participant/{datatype[:-1]}/{workerid}.png')
    plt.close()


#%%
'''
boxplots
3) for each profile group, plot the average variance of each song across participants in that profile group.
'''
pinfo_numero = load_h5file('pinfo2_numero.h5')

#%%
def get_var_per_song_given_exps(exps):
    variance_list = []
    for songurl in util.songlist:
        labels, participant_list = extract_exp_by_song_w_participant(exps, songurl, datatype)
        if len(participant_list)>0:
            variances_per_participant = np.var(labels, axis=1)
            variance_list.append(variances_per_participant)
        else:
            variance_list.append([])
    return variance_list


#%%
'''
for no profile group, all data.
'''
def single_var_list_plot(variance_list):
    fig = plt.figure(figsize=(32,8))

    plt.boxplot(variance_list)

    plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
    plt.ylim(-0.02,0.8)
    plt.ylabel(f'variance of {datatype[:-1]}')
    plt.xlabel('song')
    plt.tick_params(axis='x', which='major', labelrotation=45)
    num_participants = len(exps['workerid'].unique())
    plt.title(f'{datatype[:-1]} variances across participants for each song || {num_participants} participants || {len(exps.index)} trials')
    
    return plt

var_list = get_var_per_song_given_exps(exps)
plot = single_var_list_plot(var_list)
plot.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/alldata.png')
# plot.show()

#%%
'''
for master profile group
2 groups, subplot(221)
'''
master_exps = exps.loc[(exps['batch'] == '4')|(exps['batch']=='5')|(exps['batch']=='6')]
master_var = get_var_per_song_given_exps(master_exps)
plot = single_var_list_plot(master_var)
plot.show()

#%%

non_master_exps = exps.loc[(exps['batch'] == '7')|(exps['batch']=='8')]
non_master_var = get_var_per_song_given_exps(non_master_exps)
plot = single_var_list_plot(non_master_var)
plot.show()
#%%
fig = plt.figure(figsize=(32,12))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.92)

ax1 = plt.subplot(211)
ax1.boxplot(master_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
num_participants = len(master_exps['workerid'].unique())
plt.title(f'group 1: master participants || {num_participants} participants || {len(master_exps.index)} trials', loc='left')

ax2 = plt.subplot(212)
plt.boxplot(non_master_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
num_participants = len(non_master_exps['workerid'].unique())
plt.title(f'group 2: non master participants || {num_participants} participants || {len(non_master_exps.index)} trials', loc='left')

plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax2.tick_params(axis='x', which='major', labelrotation=45)
plt.xlabel('song')
plt.setp(ax1.get_xticklabels(), visible=False)

plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/master.png')
# plt.show()

#%%
'''
for age profile groups. 
4 groups
range_map = {(0,20):-1, (21,30):-0.33, (31,40):0.33, (41,80):1}
'''
age_groups = pinfo_numero['age'].unique()

group1_wids = pinfo_numero.loc[(pinfo_numero['age'] == age_groups[0]), ['workerid']]

group1_exps = exps.loc[exps['workerid'].isin(group1_wids.iloc[:,0].tolist())]

group2_wids = pinfo_numero.loc[(pinfo_numero['age'] == age_groups[1]), ['workerid']]

group2_exps = exps.loc[exps['workerid'].isin(group2_wids.iloc[:,0].tolist())]

group3_wids = pinfo_numero.loc[(pinfo_numero['age'] == age_groups[2]), ['workerid']]

group3_exps = exps.loc[exps['workerid'].isin(group3_wids.iloc[:,0].tolist())]

group4_wids = pinfo_numero.loc[(pinfo_numero['age'] == age_groups[3]), ['workerid']]

group4_exps = exps.loc[exps['workerid'].isin(group4_wids.iloc[:,0].tolist())]

#%%
fig = plt.figure(figsize=(32,24))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.9)

group1_var = get_var_per_song_given_exps(group1_exps)
group2_var = get_var_per_song_given_exps(group2_exps)
group3_var = get_var_per_song_given_exps(group3_exps)
group4_var = get_var_per_song_given_exps(group4_exps)

ax1 = plt.subplot(411)
ax1.boxplot(group1_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 1: 0 to 20 years of age || {len(group1_wids.index)} participants || {len(group1_exps.index)} trials', loc='left')

ax2 = plt.subplot(412)
ax2.boxplot(group2_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 2: 21 to 30 years of age || {len(group2_wids.index)} participants || {len(group2_exps.index)} trials', loc='left')

ax3 = plt.subplot(413)
ax3.boxplot(group3_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 3: 31 to 40 years of age || {len(group3_wids.index)} participants || {len(group3_exps.index)} trials', loc='left')

ax4 = plt.subplot(414)
ax4.boxplot(group4_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 4: 41 to 80 years of age || {len(group4_wids.index)} participants || {len(group4_exps.index)} trials', loc='left')

plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax4.tick_params(axis='x', which='major', labelrotation=45)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)
plt.xlabel('song')
# plt.show()
plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/age.png')

#%%
'''
country_enculturation profile group
mapper = {'US': 1, 'IN': 0, 'Others': -1}
'''
groups = pinfo_numero['country_enculturation'].unique()
groups = np.sort(groups)[::-1]

group1_wids = pinfo_numero.loc[(pinfo_numero['country_enculturation'] == groups[0]), ['workerid']]

group1_exps = exps.loc[exps['workerid'].isin(group1_wids.iloc[:,0].tolist())]

group2_wids = pinfo_numero.loc[(pinfo_numero['country_enculturation'] == groups[1]), ['workerid']]

group2_exps = exps.loc[exps['workerid'].isin(group2_wids.iloc[:,0].tolist())]

group3_wids = pinfo_numero.loc[(pinfo_numero['country_enculturation'] == groups[2]), ['workerid']]

group3_exps = exps.loc[exps['workerid'].isin(group3_wids.iloc[:,0].tolist())]


#%%

fig = plt.figure(figsize=(32,18))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.92)

group1_var = get_var_per_song_given_exps(group1_exps)
group2_var = get_var_per_song_given_exps(group2_exps)
group3_var = get_var_per_song_given_exps(group3_exps)


ax1 = plt.subplot(311)
ax1.boxplot(group1_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 1: encultured in the US || {len(group1_wids.index)} participants || {len(group1_exps.index)} trials', loc='left')

ax2 = plt.subplot(312)
ax2.boxplot(group2_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 2: encultured in India || {len(group2_wids.index)} participants || {len(group2_exps.index)} trials', loc='left')

ax3 = plt.subplot(313)
ax3.boxplot(group3_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 3: encultured elsewhere || {len(group3_wids.index)} participants || {len(group3_exps.index)} trials', loc='left')

plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax3.tick_params(axis='x', which='major', labelrotation=45)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.xlabel('song')

# plt.show()
plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/country_enculturation.png')
#%%
'''
country_live profile group
mapper = {'US': 1, 'IN': 0, 'Others': -1}
'''
groups = pinfo_numero['country_live'].unique()
groups = np.sort(groups)[::-1]

group1_wids = pinfo_numero.loc[(pinfo_numero['country_live'] == groups[0]), ['workerid']]

group1_exps = exps.loc[exps['workerid'].isin(group1_wids.iloc[:,0].tolist())]

group2_wids = pinfo_numero.loc[(pinfo_numero['country_live'] == groups[1]), ['workerid']]

group2_exps = exps.loc[exps['workerid'].isin(group2_wids.iloc[:,0].tolist())]

group3_wids = pinfo_numero.loc[(pinfo_numero['country_live'] == groups[2]), ['workerid']]

group3_exps = exps.loc[exps['workerid'].isin(group3_wids.iloc[:,0].tolist())]

#%%

fig = plt.figure(figsize=(32,18))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.92)

group1_var = get_var_per_song_given_exps(group1_exps)
group2_var = get_var_per_song_given_exps(group2_exps)
group3_var = get_var_per_song_given_exps(group3_exps)


ax1 = plt.subplot(311)
ax1.boxplot(group1_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 1: lives in the US || {len(group1_wids.index)} participants || {len(group1_exps.index)} trials', loc='left')

ax2 = plt.subplot(312)
ax2.boxplot(group2_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 2: lives in India || {len(group2_wids.index)} participants || {len(group2_exps.index)} trials', loc='left')

ax3 = plt.subplot(313)
ax3.boxplot(group3_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 3: lives elsewhere || {len(group3_wids.index)} participants || {len(group3_exps.index)} trials', loc='left')

plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax3.tick_params(axis='x', which='major', labelrotation=45)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.xlabel('song')

# plt.show()
plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/country_live.png')
#%%
'''
fav_music_lang profile group
mapper = {'EN': 1, 'TA': 0, 'Others': -1}
'''
groups = pinfo_numero['fav_music_lang'].unique()
groups = np.sort(groups)[::-1]

group1_wids = pinfo_numero.loc[(pinfo_numero['fav_music_lang'] == groups[0]), ['workerid']]

group1_exps = exps.loc[exps['workerid'].isin(group1_wids.iloc[:,0].tolist())]

group2_wids = pinfo_numero.loc[(pinfo_numero['fav_music_lang'] == groups[1]), ['workerid']]

group2_exps = exps.loc[exps['workerid'].isin(group2_wids.iloc[:,0].tolist())]

group3_wids = pinfo_numero.loc[(pinfo_numero['fav_music_lang'] == groups[2]), ['workerid']]

group3_exps = exps.loc[exps['workerid'].isin(group3_wids.iloc[:,0].tolist())]

#%%

fig = plt.figure(figsize=(32,18))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.92)

group1_var = get_var_per_song_given_exps(group1_exps)
group2_var = get_var_per_song_given_exps(group2_exps)
group3_var = get_var_per_song_given_exps(group3_exps)


ax1 = plt.subplot(311)
ax1.boxplot(group1_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 1: likes English music || {len(group1_wids.index)} participants || {len(group1_exps.index)} trials', loc='left')

ax2 = plt.subplot(312)
ax2.boxplot(group2_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 2: likes Tamil music || {len(group2_wids.index)} participants || {len(group2_exps.index)} trials', loc='left')

ax3 = plt.subplot(313)
ax3.boxplot(group3_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 3: likes other language music || {len(group3_wids.index)} participants || {len(group3_exps.index)} trials', loc='left')

plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax3.tick_params(axis='x', which='major', labelrotation=45)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.xlabel('song')

# plt.show()
plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/fav_music_lang.png')
#%%
'''
gender profile group
mapper = {'Male': 1, 'Other': 0, 'Female': -1}
'''
groups = pinfo_numero['gender'].unique()
groups = np.sort(groups)[::-1]

group1_wids = pinfo_numero.loc[(pinfo_numero['gender'] == groups[0]), ['workerid']]

group1_exps = exps.loc[exps['workerid'].isin(group1_wids.iloc[:,0].tolist())]

group2_wids = pinfo_numero.loc[(pinfo_numero['gender'] == groups[1]), ['workerid']]

group2_exps = exps.loc[exps['workerid'].isin(group2_wids.iloc[:,0].tolist())]

group3_wids = pinfo_numero.loc[(pinfo_numero['gender'] == groups[2]), ['workerid']]

group3_exps = exps.loc[exps['workerid'].isin(group3_wids.iloc[:,0].tolist())]

#%%

fig = plt.figure(figsize=(32,18))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.92)

group1_var = get_var_per_song_given_exps(group1_exps)
group2_var = get_var_per_song_given_exps(group2_exps)
group3_var = get_var_per_song_given_exps(group3_exps)


ax1 = plt.subplot(311)
ax1.boxplot(group1_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 1: Male || {len(group1_wids.index)} participants || {len(group1_exps.index)} trials', loc='left')

ax2 = plt.subplot(312)
ax2.boxplot(group2_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 2: Female || {len(group2_wids.index)} participants || {len(group2_exps.index)} trials', loc='left')

ax3 = plt.subplot(313)
ax3.boxplot(group3_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 3: Other || {len(group3_wids.index)} participants || {len(group3_exps.index)} trials', loc='left')

plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax3.tick_params(axis='x', which='major', labelrotation=45)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.xlabel('song')

# plt.show()
plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/gender.png')
#%%
'''
fav_genre profile group
mapper = {'Rock': 1, 'Pop': 0.33, 'Classical': -0.33, 'Others': -1}
'''
groups = pinfo_numero['fav_genre'].unique()
groups = np.sort(groups)[::-1]

group1_wids = pinfo_numero.loc[(pinfo_numero['fav_genre'] == groups[0]), ['workerid']]

group1_exps = exps.loc[exps['workerid'].isin(group1_wids.iloc[:,0].tolist())]

group2_wids = pinfo_numero.loc[(pinfo_numero['fav_genre'] == groups[1]), ['workerid']]

group2_exps = exps.loc[exps['workerid'].isin(group2_wids.iloc[:,0].tolist())]

group3_wids = pinfo_numero.loc[(pinfo_numero['fav_genre'] == groups[2]), ['workerid']]

group3_exps = exps.loc[exps['workerid'].isin(group3_wids.iloc[:,0].tolist())]

group4_wids = pinfo_numero.loc[(pinfo_numero['fav_genre'] == groups[3]), ['workerid']]

group4_exps = exps.loc[exps['workerid'].isin(group4_wids.iloc[:,0].tolist())]

#%%

fig = plt.figure(figsize=(32,24))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.91)

group1_var = get_var_per_song_given_exps(group1_exps)
group2_var = get_var_per_song_given_exps(group2_exps)
group3_var = get_var_per_song_given_exps(group3_exps)
group4_var = get_var_per_song_given_exps(group4_exps)

ax1 = plt.subplot(411)
ax1.boxplot(group1_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 1: likes Rock music || {len(group1_wids.index)} participants || {len(group1_exps.index)} trials', loc='left')

ax2 = plt.subplot(412)
ax2.boxplot(group2_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 2: likes Pop music || {len(group2_wids.index)} participants || {len(group2_exps.index)} trials', loc='left')

ax3 = plt.subplot(413)
ax3.boxplot(group3_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 3: likes Classical music || {len(group3_wids.index)} participants || {len(group3_exps.index)} trials', loc='left')

ax4 = plt.subplot(414)
ax4.boxplot(group4_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 4: likes other genres of music || {len(group4_wids.index)} participants || {len(group4_exps.index)} trials', loc='left')

plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax4.tick_params(axis='x', which='major', labelrotation=45)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.setp(ax3.get_xticklabels(), visible=False)

# plt.show()
plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/fav_genre.png')
#%%
'''
play_instrument profile group
mapper = {'Yes': 1, 'No': 0}
'''
groups = pinfo_numero['play_instrument'].unique()
groups = np.sort(groups)[::-1]

group1_wids = pinfo_numero.loc[(pinfo_numero['play_instrument'] == groups[0]), ['workerid']]

group1_exps = exps.loc[exps['workerid'].isin(group1_wids.iloc[:,0].tolist())]

group2_wids = pinfo_numero.loc[(pinfo_numero['play_instrument'] == groups[1]), ['workerid']]

group2_exps = exps.loc[exps['workerid'].isin(group2_wids.iloc[:,0].tolist())]


#%%

fig = plt.figure(figsize=(32,12))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.92)

group1_var = get_var_per_song_given_exps(group1_exps)
group2_var = get_var_per_song_given_exps(group2_exps)

ax1 = plt.subplot(211)
ax1.boxplot(group1_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 1: actively plays an instrument || {len(group1_wids.index)} participants || {len(group1_exps.index)} trials', loc='left')

ax2 = plt.subplot(212)
ax2.boxplot(group2_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 2: does not play an instrument actively || {len(group2_wids.index)} participants || {len(group2_exps.index)} trials', loc='left')


plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax2.tick_params(axis='x', which='major', labelrotation=45)

plt.setp(ax1.get_xticklabels(), visible=False)

plt.xlabel('song')

# plt.show()
plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/play_instrument.png')
#%%
'''
training profile group
mapper = {'Yes': 1, 'No': 0}
'''
groups = pinfo_numero['training'].unique()
groups = np.sort(groups)[::-1]

group1_wids = pinfo_numero.loc[(pinfo_numero['training'] == groups[0]), ['workerid']]

group1_exps = exps.loc[exps['workerid'].isin(group1_wids.iloc[:,0].tolist())]

group2_wids = pinfo_numero.loc[(pinfo_numero['training'] == groups[1]), ['workerid']]

group2_exps = exps.loc[exps['workerid'].isin(group2_wids.iloc[:,0].tolist())]


#%%

fig = plt.figure(figsize=(32,12))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.92)

group1_var = get_var_per_song_given_exps(group1_exps)
group2_var = get_var_per_song_given_exps(group2_exps)

ax1 = plt.subplot(211)
ax1.boxplot(group1_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 1: received musical training || {len(group1_wids.index)} participants || {len(group1_exps.index)} trials', loc='left')

ax2 = plt.subplot(212)
ax2.boxplot(group2_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 2: have not received musical training || {len(group2_wids.index)} participants || {len(group2_exps.index)} trials', loc='left')


plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax2.tick_params(axis='x', which='major', labelrotation=45)

plt.setp(ax1.get_xticklabels(), visible=False)

plt.xlabel('song')

# plt.show()
plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/training.png')
#%%
'''
training_duration profile group
range_map = {(0,0):-1, (1,5):0, (6,50):1}
'''
groups = pinfo_numero['training_duration'].unique()
groups = np.sort(groups)[::-1]

group1_wids = pinfo_numero.loc[(pinfo_numero['training_duration'] == groups[0]), ['workerid']]

group1_exps = exps.loc[exps['workerid'].isin(group1_wids.iloc[:,0].tolist())]

group2_wids = pinfo_numero.loc[(pinfo_numero['training_duration'] == groups[1]), ['workerid']]

group2_exps = exps.loc[exps['workerid'].isin(group2_wids.iloc[:,0].tolist())]

group3_wids = pinfo_numero.loc[(pinfo_numero['training_duration'] == groups[2]), ['workerid']]

group3_exps = exps.loc[exps['workerid'].isin(group3_wids.iloc[:,0].tolist())]

#%%

fig = plt.figure(figsize=(32,18))
fig.suptitle(f'{datatype[:-1]} variances across participants for each song', y=0.92)

group1_var = get_var_per_song_given_exps(group1_exps)
group2_var = get_var_per_song_given_exps(group2_exps)
group3_var = get_var_per_song_given_exps(group3_exps)


ax1 = plt.subplot(311)
ax1.boxplot(group1_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 1: received musical training for more than 5 years || {len(group1_wids.index)} participants || {len(group1_exps.index)} trials', loc='left')

ax2 = plt.subplot(312)
ax2.boxplot(group2_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 2: received musical training for 1 to 5 years || {len(group2_wids.index)} participants || {len(group2_exps.index)} trials', loc='left')

ax3 = plt.subplot(313)
ax3.boxplot(group3_var)
plt.ylim(-0.02,0.8)
plt.ylabel(f'variance of {datatype[:-1]}')
plt.title(f'group 3: have not received musical training|| {len(group3_wids.index)} participants || {len(group3_exps.index)} trials', loc='left')

plt.xticks(ticks=np.arange(1, 1+len(util.songlist)), labels=util.songlist)
ax3.tick_params(axis='x', which='major', labelrotation=45)

plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)
plt.xlabel('song')

# plt.show()
plt.savefig(f'../analysis_plots/boxplot_avevar_per_profile/{datatype[:-1]}/training_duration.png')
#%%

# %%
'''
Kat:
Thanks, Balu! Yes, and of course the analysis approach is dictated by the research question(s). I think the following are interesting:
- What is the average rating (across participants) for every time point for every song? 
(This is really the main research question, right? What are the mean arousal/valence ratings for every time point of each song across participants)
- For each of the mean ratings computed above, what is the variance (across participants)? 
This tells us how much agreement there is across participants in terms of their momentary A/V ratings for every time point in every song.

And then we can also check the following to make sure the ratings are high quality:
- How much variability is present within a single rater? (Is there low or high variance within a rater for a particular song?). 
As Balu says, one approach would be to compare the rating distribution of one individual against a reference, such as Deam reference ratings. 
Another approach would be to compare the individual rater with all of the other raters... 
this reminds me of the statistical concept called inter-rater reliability: https://en.wikipedia.org/wiki/Inter-rater_reliability , 
although I've only computed this for discrete ratings in the past, not continuous ratings like we have, 
so I'd have to see how to implement continuous IRRs. 
Something like intraclass correlation coefficients (ICC) might be useful: https://en.wikipedia.org/wiki/Intraclass_correlation

'''

'''
Dorien:
For each song, for each point in time, visualise the variance within participants. So we have an indication as to how much they agree or not. 

'''

'''
Balu:
Before that, with respect to the box plot, I am agreeing with Kat's remark: the box-plot can be used to show either 
the variation present in the ratings for a particular song for a particular reviewer (i.e., intra (or within) reviewer rating variation) 
or it can be used to show the variation present in the ratings for a particular song at a given time for multiple reviewers 
(i.e., inter (or between) reviewers rating variation). 
The former will capture idiosyncrasy of a particular reviewer and the latter give some idea about how closer are those idiosyncrasies.
.
My thought was to use such box-plot to find consistency in ratings of a reviewer and in order to achieve this we need to 
compare the rating distribution of a reviewer against a reference (in this case the Deam reference songs) and see if they are very different. 

'''
#%%

'''
misc:
1) workerid_dict. for plotting, workerids are too long. make them p1, p2, p3, p4, p5 ... etc. 

time plots
1) each song, per time, average value (-1,1 range) x: time, y: valence/arousal (+- standard deviation)
2) each song, per time stamp, variance (0,0.5 range (probably)) x: time, y: valence variance/arousal variance 

boxplots
1) for each song (non deam), average label of each participant 
2) for each participant, average label of each song 
(by mistake... i did label average rather than variance average... but i guess it's still usefull?)
1) for each song (non deam), average variance of each participant 
2) for each participant, average variance of each song
3) for each profile group, plot the average variance of each song across participants in that profile group.
'''