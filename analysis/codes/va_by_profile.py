# %%

import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 300
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"


exps = pd.read_pickle('../../data/exps_ready3.pkl')
pinfo = pd.read_pickle('../../data/mediumrare/semipruned_pinfo.pkl')
pinfo_ori = pd.read_pickle('../../data/mediumrare/unpruned_pinfo.pkl')
pinfo_n = pd.read_pickle('../../data/pinfo_numero.pkl')

# %%

affect_type = 'arousals'
conditions = ["age", "gender"]
exps = ave_exps_by_profile(exps, pinfo, affect_type, conditions)

# %%
exps_deam = exps[exps['songurl'].str.contains('deam')].reset_index()
exps_00 = exps[exps['songurl'].str.contains('00_')].reset_index()
exps_01 = exps[exps['songurl'].str.contains('01_')].reset_index()
exps_55 = exps[exps['songurl'].str.contains('0505_')].reset_index()
exps_10 = exps[exps['songurl'].str.contains('10_')].reset_index()
exps_11 = exps[exps['songurl'].str.contains('11_')].reset_index()
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

# a, b = mean_std_of_profile('arousals', exps, pinfo_n, 'gender')



# %%
# code for drawing a vertical line for segregation purposes...
# plt.axvline(x, color = 'r', linestyle='--')
nondeamexps = exps[~exps['songurl'].str.contains('deam')].reset_index()
counts = nondeamexps.groupby(['songurl']).size()
print(f'min: {counts.min()}, max: {counts.max()}, mean: {counts.mean()}')
print(counts.nlargest(5))
# min: 29, max: 64, mean: 47.32
# max is from 10_216

# %%

def get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, condition, anOrder=None):
    trials = exps[exps['songurl']==songurl].reset_index()

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
    ax.boxplot(means, labels=xticklabels+xticklabels)
    ax.set_ylim((-1,1))
    midlinepos = len(xticklabels) + 0.5
    ax.axvline([midlinepos], color = 'black', linestyle='--')
    ax.set_title('arousals', loc='left')
    ax.set_title('valences', loc='right')
    
    if standalone:
        ax.set_ylabel('mean')
        ax.set_title(f'{profilelabel} || {songurl}', y=1.07)
    else:
        ax.set_title(profilelabel, y=1.07)
        
    # return ax

# %%
songurl = 'deam_115'
# songurl = '10_216'
# songurl = '11_209'

profilelabel = 'Country of residence'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'country_live', [2,1,0])
xticklabels = ['USA', 'India', 'Others']
fig, ax = plt.subplots(figsize=(6,4))
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)

# plt.savefig(f'../plots/profile_mean_std/country_live_{songurl}_mean.png')
# plt.close()


# %%
songurl = 'deam_115'
# songurl = '10_216'
# songurl = '11_209'

profilelabel = 'Received formal musical training'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'training', [1,0])
xticklabels = ['Yes', 'No']
fig, ax = plt.subplots()
plot_mean_songurl_profile(ax, means, xticklabels, profilelabel, True, songurl)
# plt.savefig(f'../plots/profile_mean_std/training_{songurl}_mean.png')

# %%
# fig, ax = plt.subplots()
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
songurl = 'deam_115'
profilelabel = 'Country of residence'
means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'country_live', [2,1,0])
xticklabels = ['USA', 'India', 'Others']
plot_mean_songurl_profile(ax1, means, xticklabels, profilelabel, standalone=False)


means, stds = get_mean_std_given_songurl_profile(exps, pinfo_n, songurl, 'training', [1,0])
xticklabels = ['Yes', 'No']
profilelabel = 'Received formal musical training'
plot_mean_songurl_profile(ax2, means, xticklabels, profilelabel, standalone=False)
ax1.set_ylabel('mean')
fig.suptitle('deam_115',y=1.07)


# %%

df = pd.DataFrame({
    'Name': ['age', 'gender', 'residence', 'enculturation', 'language', 'genre', 'instrument', 'training', 'duration', 'master'],
    'a': [146,158,146,146,200,112,127,159,134,128],
    'b': [67,118,118,113,50,39,150,118,118,149],
    'c': [37,0,0,0,0,38,0,0,0,0],
    'd': [27,0,0,0,0,0,0,0,0,0],
    'other': [0,1,13,18,27,88,0,0,25,0],
    'extra': [0,0,0,0,0,0,0,0,0,0]
    # 'extra2': [0,0,0,0,0,0,0,0,0,0]
})

df_labels = pd.DataFrame({
    'Name': ['age', 'gender', 'residence', 'enculturation', 'language', 'genre', 'instrument', 'training', 'duration', 'master'],
    'a': ['youth', 'male', 'USA', 'USA', 'English', 'rock', 'yes', 'yes', '0 years', 'yes'],
    'b': ['young adult', 'female', 'India', 'India', 'Tamil', 'pop', 'no', 'no', '1~5', 'no'],
    'c': ['adult','','','','','classical','','','',''],
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
