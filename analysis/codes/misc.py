#%%
import os
import sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))

import util
from processing.ave_exp_by_prof import ave_exps_by_profile

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 300
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

feat_dict = util.load_pickle('../../data/feat_dict_ready2.pkl')
exps = pd.read_pickle(os.path.join('..','..','data', 'exps_ready3.pkl'))
pinfo = util.load_pickle('../../data/pinfo_numero.pkl')
# %%
arousals = exps['arousals']
# %%
count = 0
for index, row in arousals.iteritems():
    count += len(row)
    
print(count) # 783498
# %%
# count should be the same as feat_dict count since we average all trials of the same song.
exps_ave = pd.read_pickle(os.path.join('..','data', 'exps_std_a_ave3.pkl'))
aved_count = 0
for index, row in exps_ave.iterrows():
    aved_count += len(row['labels'])
print(aved_count)

feat_count = 0

for k,v in feat_dict.items():
    feat_count += len(v)
print(feat_count)

# %%
affect_type = 'arousals'
profile = ['age', 'gender']
# profile = ['age']


exps_prof_aved = ave_exps_by_profile(exps, pinfo, affect_type, profile)
prof_aved_count = 0
for index, row in exps_prof_aved.iterrows():
    prof_aved_count += len(row['labels'])
print(prof_aved_count)
# %%
'''
PLOT V-A OF ALL DEAM SONGS. 
not many high V low A songs, so... the one we use is pretty much... high V high A...S
'''
x = []
y = []

datatypes = ['arousals', 'valences']
deampath = '../../data/deam_annotations/annotations_averaged_per_song/dynamic_per_second_annotations'

arousals = pd.read_csv(os.path.join(deampath, 'arousal.csv'), index_col=None, header=0)
valences = pd.read_csv(os.path.join(deampath, 'valence.csv'), index_col=None, header=0)

for idx in np.arange(len(arousals)):
    a_mean = arousals.iloc[idx, 1:].mean()
    v_mean = valences.iloc[idx, 1:].mean()
    x.append(a_mean)
    y.append(v_mean)
    
plt.scatter(x,y,marker='.')
plt.xlabel('arousal')
plt.ylabel('valence')
plt.title('all deam songs')

# %%
# plot num participants who have labelled each song.
nondeamexps = exps[~exps['songurl'].str.contains('deam')].reset_index()

# non master
counts = nondeamexps.groupby(['songurl']).size()
fig = plt.figure(figsize=(10,4))
counts.plot(kind='bar',alpha=1, rot=80, fontsize=10)
plt.ylabel('number of participants')
plt.tight_layout()
plt.savefig('../plots/pinfo/num_p_per_song.png')

# master
pinfo = pinfo[pinfo['master'] == 1.0]
nondeamexps = nondeamexps[nondeamexps['workerid'].isin(pinfo['workerid'].unique())]
counts = nondeamexps.groupby(['songurl']).size()
fig = plt.figure(figsize=(10,4))
counts.plot(kind='bar',alpha=1, rot=80, fontsize=10)
plt.ylabel('number of participants')
plt.tight_layout()
plt.savefig('../plots/pinfo/master_num_p_per_song.png')
# %%
