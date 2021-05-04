# %%
 
import os
import glob
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../'))
import util

# %%
## 1) load feat_dict
feat_dict = util.load_pickle('../data/feat_dict_ready2.pkl')

# %%

def gather_dict_values_to_list(dictionary):
    values = list(dictionary.values())
    l = []
    for i in values:

        for j in i:
            l.append(np.array(j))
    l = np.array(l)
    return l

def reverse_dict_values_to_list(feat_dict, feat_list):
    len_dict = {e1:len(e2) for e1, e2 in feat_dict.items()}
    feats = {}
    i = 0
    for songurl, songlen in len_dict.items():
        feats[songurl] = feat_list[i:i+songlen]
        i = i+songlen
        # print(i)
    # check
    temp = {e1:len(e2) for e1, e2 in feats.items()}
    print(len_dict == temp)
    return feats

# %%
# from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

def data_transform_v(trainlist, validlist, testlist):
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    train_dict = dict((songurl, feat_dict[songurl]) for songurl in trainlist)
    valid_dict = dict((songurl, feat_dict[songurl]) for songurl in validlist)
    test_dict = dict((songurl, feat_dict[songurl]) for songurl in testlist)

    train_feats_list = gather_dict_values_to_list(train_dict)
    valid_feats_list = gather_dict_values_to_list(valid_dict)
    test_feats_list = gather_dict_values_to_list(test_dict)

    # Fit on training set only.
    scaler.fit(train_feats_list)

    # Apply transform to both the training set and the test set.
    train_img = scaler.transform(train_feats_list)
    valid_img = scaler.transform(valid_feats_list)
    test_img = scaler.transform(test_feats_list)

    return train_dict, valid_dict, test_dict, train_img, valid_img, test_img

def data_transform(trainlist, testlist):
    # scaler = StandardScaler()
    scaler = MinMaxScaler()

    train_dict = dict((songurl, feat_dict[songurl]) for songurl in trainlist)
    test_dict = dict((songurl, feat_dict[songurl]) for songurl in testlist)

    train_feats_list = gather_dict_values_to_list(train_dict)
    test_feats_list = gather_dict_values_to_list(test_dict)

    # Fit on training set only.
    scaler.fit(train_feats_list)

    # Apply transform to both the training set and the test set.
    train_img = scaler.transform(train_feats_list)
    test_img = scaler.transform(test_feats_list)

    return train_dict, test_dict, train_img, test_img

# %%
train_dict, valid_dict, test_dict, train_img, valid_img, test_img = data_transform_v(util.trainlist, util.validlist, util.testlist)
train_data = reverse_dict_values_to_list(train_dict, train_img)
valid_data = reverse_dict_values_to_list(valid_dict, valid_img)
test_data = reverse_dict_values_to_list(test_dict, test_img)

# %%


util.save_pickle('../data/train_feats.pkl', train_data)
util.save_pickle('../data/valid_feats.pkl', valid_data)
util.save_pickle('../data/test_feats.pkl', test_data)

# %%

#%%
'''
I need to do this same thing but for folds. manually will do. 5 folds. 
'''

for i in range(len(util.folds)):
    testlist = util.folds[i]
    trainlist = np.setdiff1d(util.songlist, util.folds[i])
    print(trainlist)
    print(testlist)
    train_dict, test_dict, train_img, test_img = data_transform(trainlist, testlist)
    # train_pca, test_pca = apply_pca(train_img, test_img)

    train_data = reverse_dict_values_to_list(train_dict, train_img)
    test_data = reverse_dict_values_to_list(test_dict, test_img)

    util.save_pickle(f'../data/folds/train_feats_{i}.pkl', train_data)
    util.save_pickle(f'../data/folds/test_feats_{i}.pkl', test_data)


# %%
