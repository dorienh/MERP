'''
window and partitian the features and averaged labels. leave the labels unaveraged? hmm..
'''

# %%
import os
import sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

import util


lstm_size = 10
step_size = 1




# %%
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
'''
window the features first. 
'''

def windowing(data, lstm_size, step_size):
    
    windows = []

    numwindows = len(data) - (lstm_size - step_size)

    for ts in np.arange(numwindows, step=step_size):
        window = data[ts:ts+lstm_size]
        windows.append(window)
    windows = np.array(windows)

    return windows


# %%

# feat_dict = util.load_pickle('../data/test_feats_pca.pkl')
feat_dict = util.load_pickle('../data/train_feats_pca.pkl')

dictofwindows = {}

for key, value in feat_dict.items():
    windows = windowing(value, lstm_size, step_size)
    # print(windows.shape)
    dictofwindows[key] = windows

# util.save_pickle('../data/test_feats_pca_windowed.pkl', dictofwindows)
# util.save_pickle('../data/train_feats_pca_windowed.pkl', dictofwindows)
# %%
exps = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'exps_std_a_ave.pkl'))
# exps = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'exps_std_v_ave.pkl'))

dictofwindows = {}

for songurl, labels in exps.iterrows():
    windows = windowing(np.array(labels.to_list()[0]), lstm_size, step_size)
    # print(windows.shape)
    # print(songurl)
    dictofwindows[songurl] = windows

windowed_exps = pd.DataFrame(list(zip(dictofwindows.keys(),dictofwindows.values())),  columns=['songurl', 'labels'])
windowed_exps.set_index('songurl', inplace=True)

util.save_pickle('../data/exps_std_a_ave_windowed.pkl', windowed_exps)
# util.save_pickle('../data/exps_std_v_ave_windowed.pkl', windowed_exps)

# %%
'''
so... how to reverse window for testing?
'''
temp = exps.at['deam_115', 'labels']
hoge = windowed_exps.at['deam_115', 'labels']

def reverse_windowing(data, lstm_size, step_size):
    reverse_step_size = lstm_size//step_size

    original_len = len(data) + (lstm_size-step_size)

    original_data = []

    i=0
    while i < (original_len - reverse_step_size):
        original_data.append(data[i])
        print('i: ', i)
        print(sum([len(x) for x in original_data]))
        i += reverse_step_size
    original_data.append(data[-1][(i-original_len)::])
    original_data = [item for sublist in original_data for item in sublist]
    original_data = np.array(original_data)

    return original_data

reverse_windowing(hoge, 10, 1)

# %%
