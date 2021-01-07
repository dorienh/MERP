#%% 

import pandas as pd
import numpy as np

# https://github.com/anthropedia/tci-stats/blob/master/tcistats/__init__.py
def cronbach_alpha(items):
    items = pd.DataFrame(items)
    items_count = items.shape[1]

    # print(items_count)
    variance_sum = float(items.var(axis=0, ddof=1).sum())
    total_var = float(items.sum(axis=1).var(ddof=1))
    
    return (items_count / float(items_count - 1) *
            (1 - variance_sum / total_var))

# %%
# https://towardsdatascience.com/cronbachs-alpha-theory-and-application-in-python-d2915dd63586
def cronbach_alpha(df):
    # 1. Transform the df into a correlation matrix
    df_corr = df.corr()
    # print(df_corr.shape)
    # 2.1 Calculate N
    # The number of variables equals the number of columns in the df
    N = df.shape[1]
    
    # 2.2 Calculate R
    # For this, we'll loop through the columns and append every
    # relevant correlation to an array calles "r_s". Then, we'll
    # calculate the mean of "r_s"
    rs = np.array([])
    for i, col in enumerate(df_corr.columns):
        sum_ = df_corr[col][i+1:].values
        rs = np.append(sum_, rs)
    mean_r = np.mean(rs)
    
   # 3. Use the formula to calculate Cronbach's Alpha 
    cronbach_alpha = (N * mean_r) / (1 + (N - 1) * mean_r)
    return cronbach_alpha
#%%
temp = exps.groupby('songurl').get_group('00_145')
temp = temp['arousals']
temp1 = np.array(temp.tolist()).T

corrmat = np.corrcoef(temp1)
#%%
labels_df = pd.DataFrame(temp.values.tolist())

cronbach_alpha(labels_df)

# %%
# find the cronbach alpha for all songs, then their mean and std.
def cronbach_alpha_for_all_songs(datatype):
    cronalpha_list = []
    cronalpha_dict = {}
    for songurl, song_group in exps.groupby('songurl'):
        # if 'deam' not in songurl:
        #     pass
        # else:
        labels = song_group[datatype]
        labels_df = pd.DataFrame(labels.values.tolist())
        # temp = [len(a) for a in labels]
        # print(all(x==temp[0] for x in temp))
        cronbach_alpha_value = cronbach_alpha(labels_df)
        cronalpha_list.append(cronbach_alpha_value)
        cronalpha_dict[songurl] = cronbach_alpha_value
    print(cronalpha_dict)

    print(f'{np.mean(cronalpha_list):5f} +- {np.std(cronalpha_list):5f}')

    return cronalpha_dict

# %%
exps = pd.read_pickle(os.path.join(os.path.abspath('../..'), 'data', 'exps_ready.pkl'))

# datatype = 'valences'

arousal = cronbach_alpha_for_all_songs('arousals')
valence = cronbach_alpha_for_all_songs('valences')

# %%
'''
PLOT
https://matplotlib.org/3.1.1/gallery/subplots_axes_and_figures/figure_title.html
'''

import matplotlib.pyplot as plt
from cycler import cycler

# plt.rc('axes', prop_cycle=(cycler(color=['r', 'g', 'b', 'y'])))
fig, axs = plt.subplots(2,sharex=True, figsize=(14, 10), constrained_layout=True)

bar_colours = []


axs[0].bar(arousal.keys(), arousal.values())
# axs[0].xticks(rotation=90)
# axs[0].ylim((-1, 1))
axs[0].set_ylabel('Arousal')


axs[1].bar(valence.keys(), valence.values())
plt.xticks(rotation=90)
plt.ylim((0, 1))
axs[1].set_ylabel('Valence')

# plt.tight_layout()
plt.show()
# plt.savefig('../plots/interreliability/cronbachalpha.png')
# plt.savefig('../plots/interreliability/cronbachalpha.tiff')
# plt.close()


#%%
'''
https://pythonbasics.org/matplotlib-bar-chart/
'''

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(16,5))

width = 0.4
plt.bar(np.arange(len(arousal.values())), arousal.values(), width=width, label='arousal')
plt.bar(np.arange(len(valence.values()))+ width, valence.values(), width=width, label='valence')
plt.xticks(np.arange(len(arousal.keys())), arousal.keys(), rotation=90)
plt.legend()
plt.tight_layout()
# plt.show()
plt.savefig('../plots/interreliability/cronbachalpha.png')
plt.close()

# %%
