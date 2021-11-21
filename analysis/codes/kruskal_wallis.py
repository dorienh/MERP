'''
references:
https://www.statology.org/kruskal-wallis-test-python/
https://www.statology.org/dunns-test-python/
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kruskal.html
'''

# %%
from operator import sub
import pandas as pd
pd.set_option('mode.chained_assignment', None)
# https://www.dataquest.io/blog/settingwithcopywarning/
import numpy as np

from scipy import stats
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns

# %%
exps = pd.read_pickle('../../data/exps_ready3.pkl')
pinfo_numero = pd.read_pickle('../../data/pinfo_numero.pkl')


# %%
r_mapper_dict = {'age': {0.0: '(0, 25)', 0.33: '(26, 35)', 0.66: '(36, 50)', 1.0: '(51, 80)'}, 
'gender': {1.0: 'Female', 0.5: 'Other', 0.0: 'Male'}, 
'residence': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'enculturation': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'language': {1.0: 'English', 0.5: 'Tamil', 0.0: 'Other'}, 
'genre': {1.0: 'Rock', 0.66: 'Classical', 0.33: 'Pop', 0.0: 'Other'}, 
'instrument': {1.0: 'Yes', 0.0: 'No'}, 
'training': {1.0: 'Yes', 0.0: 'No'}, 
'duration': {0: '(0, 0)', 0.5: '(1, 5)', 1.0: '(6, 50)'}, 
'master': {1.0: 'Yes', 0.0: 'No'}}

pinfo = pinfo_numero.replace(r_mapper_dict)

# %%
def get_sub_exps(affect_type, profile_type, exps):
    
    sub_exps = exps[['workerid',affect_type]]
    # average the affect labels for each trial
    sub_exps[affect_type] = sub_exps[affect_type].apply(lambda x: np.mean(x))

    # map workerid to profile group
    mapping = dict(pinfo[['workerid',profile_type]].values)
    sub_exps['group'] = sub_exps.workerid.map(mapping)
    return sub_exps

def reformat_to_col_per_group(sub_exps, affect_type):
    # reformat dataframe for anova/kruskal
    temp_dict = {}
    for group in r_mapper_dict[profile_type].values():
        print(group)
        temp_dict[group] = list(sub_exps[sub_exps['group']==group][affect_type])
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in temp_dict.items() ]))
    return df


# %%

def apply_test(affect_type, profile_type, exps):
    sub_exps = get_sub_exps(affect_type, profile_type, exps)

    # reformat dataframe for anova/kruskal
    df = reformat_to_col_per_group(sub_exps, affect_type)
    # print(df.head())

    data = [df[col].dropna() for col in df]
    data = [x for x in data if not x.empty]
    kstat, pval = stats.kruskal(*data)

    print(f"kruskal wallis p value: {pval:.4f}")

    
    if pval<0.05:
    #     # print(f"reject null hypothesis that median {affect_type} is the same for all {profile_type} groups")
        
        dunn = sp.posthoc_dunn(data, p_adjust = 'bonferroni')
        print('dunn: ', dunn)
    #     # print(dunn.rename({colidx: groupname for colidx, groupname in zip(dunn.columns, list(r_mapper_dict[profile_type].values()))}))
    #     print('\n')

    #     return True
    # else:
    #     # print(f"lack of evidence to reject null hypothesis. no statistically significant difference between groups")
    #     print('\n')
    #     return False

    return float(f'{pval:.4f}')

# %%

pinfo_master = pinfo[pinfo['master'] == 'Yes']
# exps_master = exps[exps['workerid'].isin(pinfo_master['workerid'].unique())]
# exps_nonmaster = exps[~exps['workerid'].isin(pinfo_master['workerid'].unique())]

# %%
result_dict = {}
for profile_type in r_mapper_dict.keys():

    result_dict[profile_type] = {}
    print(f"# profile: {profile_type}")
    for affect_type in ['arousals', 'valences']:
        print(f"## {affect_type}")
        pval = apply_test(affect_type,profile_type,exps)

        result_dict[profile_type][affect_type] = pval
        
print(pd.DataFrame(result_dict).T)
# dunn = apply_test('valences', 'age', exps)

# plot_dist_av('residence', exps_master, filename='master')
# plot_violin_av('residence', exps_nonmaster, filename='nonmaster')
        
# %%
print(pd.DataFrame(result_dict).transpose())
# %%

def plot_dist_av(profile_type, exps, filename=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    # print(axes[1])
    for idx, affect_type in enumerate(['arousals', 'valences']):
        sub_exps = get_sub_exps(affect_type, profile_type, exps)
        for group in r_mapper_dict[profile_type].values():
            subset = sub_exps[sub_exps['group'] == group]
            sns.distplot(subset[affect_type],label=group, ax=axes[idx])

            if result_dict[profile_type][affect_type]<0.05:
                axes[idx].xaxis.label.set_color('green')

    handles, labels = axes[idx].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(profile_type)
    if filename:
        plt.savefig(f'../plots/dist_profile/{profile_type}_{filename}.png')
    else:
        plt.savefig(f'../plots/dist_profile/{profile_type}.png')


# %%
## violin plots?

sub_exps = get_sub_exps('arousals', 'age', exps)
# for group in r_mapper_dict['age'].values():
    # subset = sub_exps[sub_exps['group'] == group]
    # print(group,subset)
sns.violinplot(x="arousals", y="group", data=sub_exps)
    # break

#%%
# profile_type='genre'
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14,5))
# # print(axes[1])
# for idx, affect_type in enumerate(['arousals', 'valences']):
#     sub_exps = get_sub_exps(affect_type, profile_type, exps)
    
#     sns.violinplot(x="group", y=affect_type, data=sub_exps, ax=axes[idx])

#     if result_dict[profile_type][affect_type]<0.05:
#         axes[idx].xaxis.label.set_color('green')

# plt.show()
def plot_violin_av(profile_type, exps, filename=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    # print(axes[1])
    for idx, affect_type in enumerate(['arousals', 'valences']):
        sub_exps = get_sub_exps(affect_type, profile_type, exps)
        
        sns.violinplot(x="group", y=affect_type, data=sub_exps, ax=axes[idx])

        if result_dict[profile_type][affect_type]<0.05:
            axes[idx].xaxis.label.set_color('green')

    handles, labels = axes[idx].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(profile_type)
    if filename:
        plt.savefig(f'../plots/violin_profile/{profile_type}_{filename}.png')
    else:
        plt.savefig(f'../plots/violin_profile/{profile_type}.png')
# %%
for profile_type in r_mapper_dict.keys():
    
    print(f"# profile: {profile_type}")
    # plot_violin_av(profile_type, exps)
    plot_dist_av(profile_type, exps)
# %%
'''
double check with mann-whitney u test
'''
affect_type = 'arousals'
profile_type = 'master'
sub_exps = get_sub_exps(affect_type, profile_type, exps)
# reformat dataframe for anova/kruskal
df = reformat_to_col_per_group(sub_exps, affect_type)

data = [df[col].dropna() for col in df]
_, pval = stats.mannwhitneyu(*data[0:2])
print(f'mannwhitenyu: {pval}')

apply_test(affect_type, profile_type, exps)

# %%
# plot dist plot without averaging per song.
affect_type = 'arousals'
profile_type = 'training'

def get_sub_exps2(affect_type, profile_type, exps):

    sub_exps = exps[['workerid',affect_type]]
    mapping = dict(pinfo[['workerid',profile_type]].values)
    sub_exps['group'] = sub_exps.workerid.map(mapping)
    del sub_exps['workerid']
    # print(sub_exps.head())
    sub_exps = sub_exps.explode(affect_type, ignore_index=True)
    # print(sub_exps.head())
    return sub_exps

# df = reformat_to_col_per_group(sub_exps, affect_type)
def plot_dist_av(profile_type, result_dict):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))
    # print(axes[1])
    for idx, affect_type in enumerate(['arousals', 'valences']):
        sub_exps = get_sub_exps2(affect_type, profile_type, exps)
        for group in r_mapper_dict[profile_type].values():
            subset = sub_exps[sub_exps['group'] == group]
            sns.distplot(subset[affect_type],label=group, ax=axes[idx])

            if float(result_dict[profile_type][affect_type])<0.05:
                axes[idx].xaxis.label.set_color('green')

    handles, labels = axes[idx].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(profile_type)
    # plt.show()
    plt.savefig(f'../plots/dist_profile/not_ave_{profile_type}.png')

for profile_type in r_mapper_dict.keys():
    
    print(f"# profile: {profile_type}")
    plot_dist_av(profile_type, result_dict)


#%%
# %%
# split age groups into 10s. 

pinfo2 = pd.read_pickle('../../data/mediumrare/semipruned_pinfo.pkl')
pinfo2 = pinfo2[pinfo2['workerid'].isin(exps['workerid'].unique())]
# min age is 19 lol no worries. 

# age_map = {(0, 19): 0.0, (20, 29): 0.16, (30, 39): 0.32, (40, 49): 0.48, (50, 59): 0.64, (60, 69): 0.8, (70, 80): 1.0}
'''
0.16    133
0.32     80
0.48     38
0.64     18
0.80      8
0.00      1
1.00      1
'''
age_map = {(0, 29): 0.0, (30, 39): 0.25, (40, 49): 0.50, (50, 59): 0.75, (60, 80): 1.0}

def numerical(range_map, arr):
    def get_cat(range_map, x):
        for key in range_map:
            x = int(x)
            if key[0] <= x <= key[1]:
                return range_map[key]
    retval = [get_cat(range_map, a) for a in arr]
    # for a in arr:
        # print(a, ' || ', get_cat(range_map, a))
    return retval

n_pinfo_dict = {
    'workerid': pinfo2['workerid']
}
ocol = pinfo2['age'].to_numpy()
ncol = numerical(age_map, ocol)
n_pinfo_dict['age'] = ncol
df = pd.DataFrame(n_pinfo_dict)
print(pd.Series(ncol).value_counts())




fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12,5))

for idx, affect_type in enumerate(['arousals', 'valences']):
    sub_exps = exps[['workerid',affect_type]]
    # average the affect labels for each trial
    sub_exps[affect_type] = sub_exps[affect_type].apply(lambda x: np.mean(x))

    # map workerid to profile group
    mapping = dict(df.values)
    sub_exps['group'] = sub_exps.workerid.map(mapping)

    # df_re = reformat_to_col_per_group(sub_exps, affect_type)
    temp_dict = {}
    for group in age_map.values():
        temp_dict[group] = list(sub_exps[sub_exps['group']==group][affect_type])
    df_re = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in temp_dict.items() ]))
    data = [df_re[col].dropna() for col in df_re]
    _, pval = stats.kruskal(*data)

    _, pval2 = stats.mannwhitneyu(*data[0:2])
    print(f'mannwhitenyu: {pval2}')

    for group in age_map.values():
        subset = sub_exps[sub_exps['group'] == group]
        sns.distplot(subset[affect_type],label=group, ax=axes[idx])
        axes[idx].set_xlabel(f'{affect_type} (p = {pval:.4f})')
    if pval<0.05:
        axes[idx].xaxis.label.set_color('green')
    
        dunn = sp.posthoc_dunn(data, p_adjust = 'bonferroni')
        print('dunn')
        print(dunn)

handles, labels = axes[idx].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle('age')
plt.savefig(f'../plots/dist_profile/age10sbins.png')

# %%
# how many participants country=India and are non master?

pinfo.groupby(["residence", "master"]).size()
# %%
