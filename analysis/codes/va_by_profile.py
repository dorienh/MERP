# %%

import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
from processing.ave_exp_by_prof import ave_exps_by_profile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 300
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

import util

# exps = pd.read_pickle('../../data/exps_ready3.pkl')
# pinfo = util.load_pickle('../../data/pinfo_numero.pkl')
# pinfo = pinfo[pinfo['master'] == 1.0]
# exps = exps[exps['workerid'].isin(pinfo['workerid'].unique())]


exps = pd.read_pickle('../../data/exps_ready3.pkl')
# pinfo = pd.read_pickle('../../data/mediumrare/semipruned_pinfo.pkl')
# pinfo_ori = pd.read_pickle('../../data/mediumrare/unpruned_pinfo.pkl')
pinfo_n = pd.read_pickle('../../data/pinfo_numero.pkl')

# copy pasted from numerify_pinfo.py... (was not able to import it successfully.)
mapper_dict = {
    'age': {(0,25):0.0, (26,35):0.33, (36,50):0.66, (51,80):1.0},
    'gender': {'Female': 1.0, 'Other': 0.5, 'Male': 0.0},
    'residence': {'US': 1.0, 'IN': 0.5, 'Other': 0.0},
    'enculturation': {'US': 1.0, 'IN': 0.5, 'Other': 0.0},
    'language': {'EN': 1.0, 'TA': 0.5, 'Other': 0.0},
    'genre': {'Rock': 1.0, 'Classical music': 0.66, 'Pop': 0.33, 'Other': 0.0},
    'instrument': {'Yes': 1.0, 'No': 0.0},
    'training': {'Yes': 1.0, 'No': 0.0},
    'duration': {(0,0):0, (1,5):0.5, (6,50):1.0},
    'master': {'Yes': 1.0, 'No': 0.0}
}
# %%
# reverse mapper_dict
def reverse_nested_dict_keyval(mapper_dict):
    r_mapper_dict = {}
    for profile_type, mapper in mapper_dict.items():
        r_mapper = {}
        for key, val in mapper.items():
            r_mapper[val] = key
        r_mapper_dict[profile_type] = r_mapper
    # print(r_mapper_dict)
    return r_mapper_dict
# r_mapper_dict = reverse_nested_dict_keyval

# reverse mapper_dict manually edited.
r_mapper_dict = {'age': {0.0: (0, 25), 0.33: (26, 35), 0.66: (36, 50), 1.0: (51, 80)}, 
'gender': {1.0: 'Female', 0.5: 'Other', 0.0: 'Male'}, 
'residence': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'enculturation': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'language': {1.0: 'English', 0.5: 'Tamil', 0.0: 'Other'}, 
'genre': {1.0: 'Rock', 0.66: 'Classical', 0.33: 'Pop', 0.0: 'Other'}, 
'instrument': {1.0: 'Yes', 0.0: 'No'}, 
'training': {1.0: 'Yes', 0.0: 'No'}, 
'duration': {0: (0, 0), 0.5: (1, 5), 1.0: (6, 50)}, 
'master': {1.0: 'Yes', 0.0: 'No'}}
# %%

affect_type = 'arousals'
conditions = ["age"]
# exps = ave_exps_by_profile(exps, pinfo, affect_type, conditions)

'''
pick out the master participants only and test.
'''
master_pinfo_n = pinfo_n[pinfo_n['master'] == 1.0]
exps_master = ave_exps_by_profile(exps, master_pinfo_n, affect_type, conditions)

exps_master_np = exps[exps['workerid'].isin(master_pinfo_n['workerid'].unique())]

# %%
'''
DEAM COMPARISON
retreive DEAM data.
'''
datatypes = ['arousals', 'valences']
deampath = '../../data/deam_annotations/annotations_per_each_rater/dynamic-persec'
deamsonglist = ['115', '343', '745', '1334']

def get_deam_annotations(datatypes, deampath, deamsonglist):
    deamlabels = {datatypes[0]:{}, datatypes[1]:{}}
    for datatype in datatypes:
        for deamsong in deamsonglist:
            path = os.path.join(deampath, datatype[:-1], f'{deamsong}.csv')
            # print(path)
            deamlabels[datatype][f'deam_{deamsong}'] =  pd.read_csv(path, index_col=None, header=0) 
    return deamlabels

def get_static_deamstats(datatype, deamlabels):
    deamstats = {datatypes[0]:{}, datatypes[1]:{}}
    for datatype in deamlabels.keys():
        for songurl, df in deamlabels[datatype].items():
            ave = df.mean().mean()
            stddev = df.std().std()
            deamstats[datatype][songurl] = {'ave':round(ave,4), 'stddev':round(stddev,4)}
    return deamstats

deamlabels = get_deam_annotations(datatypes, deampath, deamsonglist)
deamstats = get_static_deamstats(datatypes, deamlabels)
print(pd.DataFrame(deamstats))

# %%
# plot deam box plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
axlist = [ax1,ax2,ax3,ax4]

for idx, deamidx in enumerate(deamsonglist):


# %%
# exps_deam = exps[exps['songurl'].str.contains('deam')].reset_index()
# exps_00 = exps[exps['songurl'].str.contains('00_')].reset_index()
# exps_01 = exps[exps['songurl'].str.contains('01_')].reset_index()
# exps_55 = exps[exps['songurl'].str.contains('0505_')].reset_index()
# exps_10 = exps[exps['songurl'].str.contains('10_')].reset_index()
# exps_11 = exps[exps['songurl'].str.contains('11_')].reset_index()
# %%
# {(0,25):0.0, (26,35):0.33, (36,50):0.66, (51,80):1.0}
# {'US': 1.0, 'IN': 0.5}
# {'EN': 1.0, 'TA': 0.5}
# {'Male': 0.0, 'Other': 0.5, 'Female': 1.0}
# {'Rock': 1.0, 'Pop': 0.33, 'Classical music': 0.66}
# {'Yes': 1.0, 'No': 0.0}
# {(0,0):0, (1,5):0.5, (6,50):1.0}

# for p_type, group in pinfo_n.groupby('gender'):
#     print(p_type)

# wids = pinfo[pinfo['gender']=='Other']['workerid']
# trials = exps[exps['workerid'].isin(wids)]

#%%
def mean_std_from_series_of_lists(aSeries):
    listofaves = []
    listofstds = []
    for _, aList in aSeries.iteritems():
        listofaves.append(aList.mean())
        listofstds.append(aList.std())
    return listofaves, listofstds

# a, b = mean_std_from_series_of_lists(trials['arousals'])

def mean_std_of_profile(affect_type, exps, pinfo_n, conditions):
    list_means = []
    list_stds = []
    for p_type, group in pinfo_n.groupby(conditions):
        wids = group['workerid']
        trials = exps[exps['workerid'].isin(wids)]
        
        means, stds = mean_std_from_series_of_lists(trials[affect_type])
        list_means.append(means)
        list_stds.append(stds)
        print(p_type, len(means))
    return list_means, list_stds

# check the order necessary for plotting stacked horizontal bar chart here. 
# typically it requires a reverse in order.
a, b = mean_std_of_profile('arousals', exps, pinfo_n, 'duration')




# %%
# check which songs are labelled by the most participants.
nondeamexps = exps[~exps['songurl'].str.contains('deam')].reset_index()
counts = nondeamexps.groupby(['songurl']).size()
print(f'min: {counts.min()}, max: {counts.max()}, mean: {counts.mean()}')
# print top n most labelled songs
print(counts.nlargest(5))
# min: 29, max: 64, mean: 47.32
# max is from 10_216

# %%

def get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, condition, anOrder=None):
    if songurl == 'all songs':
        trials = exps
    else:
        trials = exps[exps['songurl']==songurl].reset_index()
        print(f'number of trials for song {songurl}: {len(trials)}')

    a,b = mean_std_of_profile('arousals', trials, pinfo_n, condition)
    c,d = mean_std_of_profile('valences', trials, pinfo_n, condition)

    if anOrder:
        def reorder_list_elements(anOrder, aList):
            reordered_list = [aList[i] for i in anOrder]
            return reordered_list
        a = reorder_list_elements(anOrder, a)
        b = reorder_list_elements(anOrder, b)
        c = reorder_list_elements(anOrder, c)
        d = reorder_list_elements(anOrder, d)

    return a+c, b+d

def plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, standalone=True, songurl=''):
    '''
    means: expects a list of lists, arousals first then valence.
    '''
    ax.boxplot(means, labels=xticklabels+xticklabels)
    ax.set_ylim((-1,1))
    midlinepos = len(xticklabels) + 0.5
    ax.axvline([midlinepos], color = 'black', linestyle='--')
    ax.set_title('arousals', loc='left', fontsize=12)
    ax.set_title('valences', loc='right', fontsize=12)
    
    if standalone:
        ax.set_ylabel('mean')
        ax.set_title(f'{profilelabel} || {songurl}', y=1.07, fontsize=12)
    else:
        ax.set_title(profilelabel, y=1.07, fontsize=12)
        
    # return ax
def plot_deam_line(ax, deamstats, songurl):
    a_mean = deamstats['arousals'][songurl]['ave']
    v_mean = deamstats['valences'][songurl]['ave']
    ax.axhline([a_mean], xmin=0, xmax=0.5,color = 'purple', linestyle='--')
    ax.axhline([v_mean], xmin=0.5, xmax=1,color = 'purple', linestyle='--')

# %%
deamurl = 'deam_115'
# def plot_mean_songurl_profile_deam(ax, means, xticklabels, profilelable, deamlabels, deamurl, standalone=True, ):

deamlabels[deamurl]['arousals']



# %%
# deamsonglist = ['115', '343', '745', '1334']
# songurl = 'deam_115'
# songurl = '11_209'
# songurl = 'deam_1334'
songurl= 'all songs'

profilelabel = 'Country of residence'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'residence', [2,1,0])
xticklabels = ['USA', 'India', 'Others']
fig, ax = plt.subplots(figsize=(6,4))
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
# plot_deam_line(ax, deamstats, songurl)

plt.savefig(f'../plots/profile_mean_std/{songurl}_residence_mean.png')

# %%
profilelabel = 'Received formal musical training'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'training', [1,0])
xticklabels = ['Yes', 'No']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
plt.savefig(f'../plots/profile_mean_std/{songurl}_training_mean.png')

# %%
profilelabel = 'Age group'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'age')
xticklabels = ['(0, 25)', '(26, 35)', '(36, 50)', '(51, 80)']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
plt.savefig(f'../plots/profile_mean_std/{songurl}_age_mean.png')

# %%
profilelabel = 'Gender'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'gender', [0,2,1])
xticklabels = ['male', 'female', 'other']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
plt.savefig(f'../plots/profile_mean_std/{songurl}_gender_mean.png')

# plt.close()
# %%
profilelabel = 'Country of music enculturation'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'enculturation', [2,1,0])
xticklabels = ['USA', 'India', 'Others']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
plt.savefig(f'../plots/profile_mean_std/{songurl}_enculturation_mean.png')

# %%
profilelabel = 'Actively plays an instrument'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'instrument', [1,0])
xticklabels = ['Yes', 'No']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
plt.savefig(f'../plots/profile_mean_std/{songurl}_instrument_mean.png')

# %%
profilelabel = 'Duration of formal training received (years)'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'duration')
xticklabels = ['0', '1-5', '>6']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
plt.savefig(f'../plots/profile_mean_std/{songurl}_duration_mean.png')

# %%
profilelabel = 'Master participant or not'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'master', [1,0])
xticklabels = ['Yes', 'No']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
plt.savefig(f'../plots/profile_mean_std/{songurl}_master_mean.png')

# %%
profilelabel = 'Preferred language of lyrics'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'language', [2,1,0])
xticklabels = ['English', 'Tamil', 'Other']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
plt.savefig(f'../plots/profile_mean_std/{songurl}_language_mean.png')

# %%
profilelabel = 'Preferred genre'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'genre', [3,2,1,0])
xticklabels = ['Rock', 'Classical', 'Pop', 'Other']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
plt.savefig(f'../plots/profile_mean_std/{songurl}_genre_mean.png')

# %%
# just to get a feel of all the deam songs. this cell plots all 4 deam songs. 
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
axlist = [ax1,ax2,ax3,ax4]
profilelabel = 'Country of residence'
xticklabels = ['USA', 'India', 'Others']
for idx, deamidx in enumerate(deamsonglist):
    songurl = f'deam_{deamidx}'
    means, std = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'residence', [2,1,0])
    plot_mean_songurl_profile(axlist[idx], std, xticklabels, songurl, standalone=False)
    plot_deam_line(axlist[idx], deamstats, songurl)
plt.tight_layout()
plt.savefig(f'../plots/profile_mean_std/residence_alldeam_{songurl}_std.png')
# %%
# songurl = 'deam_115'
# songurl = '10_216'
# songurl = '11_209'



# %%
# fig, ax = plt.subplots()
fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,8))
# songurl1 = 'deam_115'
# songurl2 = '11_209'
songurl1 = 'deam_745' # 00
songurl2 = '00_35' 
profilelabel = 'Country of residence'
xticklabels = ['USA', 'India', 'Others']
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl1, 'residence', [2,1,0])
plot_mean_songurl_profile(ax1, means, xticklabels, profilelabel, standalone=False)
plot_deam_line(ax1, deamstats, songurl1)

means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl2, 'residence', [2,1,0])
plot_mean_songurl_profile(ax3, means, xticklabels, profilelabel, standalone=False)
ax1.set_ylabel('mean')
plt.figtext(0.5,1, songurl1, fontsize=12, ha="center", va="top", fontweight='bold')

xticklabels = ['Yes', 'No']
profilelabel = 'Received formal musical training'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl1, 'training', [1,0])
plot_mean_songurl_profile(ax2, means, xticklabels, profilelabel, standalone=False)
plot_deam_line(ax2, deamstats, songurl1)

means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl2, 'training', [1,0])
plot_mean_songurl_profile(ax4, means, xticklabels, profilelabel, standalone=False)
ax3.set_ylabel('mean')
plt.figtext(0.5,0.5, songurl2, fontsize=12, ha="center", va="top", fontweight='bold')

fig.tight_layout(h_pad=3.0)

plt.savefig(f'../plots/profile_mean_std/residence_training_{songurl2[0:2]}_stds.png')
#%%

# %%

df = pd.DataFrame({
    'Name': ['age', 'gender', 'residence', 'enculturation', 'language', 'genre', 'instrument', 'training', 'duration', 'master'],
    'a': [37,158,146,146,200,88,127,159,118,128],
    'b': [147,118,118,113,50,39,150,118,134,149],
    'c': [67,0,0,0,0,38,0,0,0,0],
    'd': [27,0,0,0,0,0,0,0,0,0],
    'other': [0,1,13,18,27,112,0,0,25,0],
    'extra': [0,0,0,0,0,0,0,0,0,0]
    # 'extra2': [0,0,0,0,0,0,0,0,0,0]
})

df_labels = pd.DataFrame({
    'Name': ['age', 'gender', 'residence', 'enculturation', 'language', 'genre', 'instrument', 'training', 'duration', 'master'],
    'a': ['youth', 'male', 'USA', 'USA', 'English', 'rock', 'yes', 'yes', '0 years', 'yes'],
    'b': ['young adult', 'female', 'India', 'India', 'Tamil', 'classical', 'no', 'no', '1~5', 'no'],
    'c': ['adult','','','','','pop','','','',''],
    'd': ['elder','','','','','','','','',''],
    'other':['', 'other', 'other', 'other', 'other', 'other', '', '', '>5', '']
})

df_total = df["a"] + df["b"] + df["c"] + df["d"] + df["other"]
df_rel = df[df.columns[1:]].div(df_total, 0)*100
df_rel['Name'] = ['age', 'gender', 'residence', 'enculturation', 'language', 'genre', 'instrument', 'training', 'duration', 'master']

ax = df_rel.plot(
  x = 'Name', 
  kind = 'barh', 
  stacked = True, width=0.8,
#   title = 'Proportion of participants by profile', 
  mark_right = True,
  legend=False,
  cmap="RdBu", alpha=0.6,
  figsize=(7,6), fontsize=12
  )
ax.set_ylabel('Profile type', fontsize=12)
ax.set_xlabel('Percentage of participants', fontsize=12)
# plt.close()
for n in df_rel.columns[:-2]:
    # print('n: ', n)

    for i, (cs, ab) in enumerate(zip(df_rel.iloc[:,:-2].cumsum(1)[n], df_rel[n])):
        if np.round(ab, 1) > 0:
            plt.text(cs - ab / 2, i, df_labels.iloc[i][n], va = 'center', ha = 'center', fontsize=12)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('../plots/pinfo/stacked_bars.png')

# %%



# %%
