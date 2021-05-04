'''
jupyter notebook for reformatting.

EXPS

exps_a_ave.pkl
exps_v_ave.pkl

exps_a.pkl
exps_v.pkl

AUDIOFEAT



'''

#%%
import os
import sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))

import util

import pandas as pd
import numpy as np


feat_dict = util.load_pickle('../data/feat_dict_ready.pkl')
exps = pd.read_pickle(os.path.join('..','data', 'exps_ready3.pkl'))
# %%
############################
####    exps average    ####
############################

def average_exps_by_songurl(exps, affect_type):
    ave_labels = {}
    for songurl, group in exps.groupby('songurl'):
        ave = group[affect_type].mean()
        ave_labels[songurl] = ave

    return ave_labels

def gather_dict_values_to_list(dictionary):
    values = list(dictionary.values())
    l = []
    for i in values:

        for j in i:
            l.append(np.array(j))
    l = np.array(l)
    return l

def reverse_dict_values_to_list(dictionary, list):
    len_dict = {e1:len(e2) for e1, e2 in dictionary.items()}
    ori_dict = {}
    i = 0
    for songurl, songlen in len_dict.items():
        ori_dict[songurl] = list[i:i+songlen]
        i = i+songlen
        # print(i)
    # check
    temp = {e1:len(e2) for e1, e2 in ori_dict.items()}
    print(len_dict == temp)
    return ori_dict

# %%

# affect_type = 'arousals'
affect_type = 'valences'

ave_exps = average_exps_by_songurl(exps, affect_type)
# all_values = gather_dict_values_to_list(ave_exps)

# mean = all_values.mean()
# std = all_values.std()

# standardized_values = (all_values - mean) / std

# standardized_ave_exps = reverse_dict_values_to_list(ave_exps, standardized_values)

# standardized_ave_exps.keys()

# exps.head()
'''
for averaged values, we don't care about wid so just have 2 columns, song url and labels.
'''
new_exps = pd.DataFrame(list(zip(ave_exps.keys(),ave_exps.values())),  columns=['songurl', 'labels'])
new_exps.set_index('songurl', inplace=True)
new_exps.head()

import pickle

with open(f'../data/exps_std_{affect_type[0]}_ave3.pkl', 'wb') as handle:
    pickle.dump(new_exps, handle, protocol=pickle.HIGHEST_PROTOCOL)
# %%
