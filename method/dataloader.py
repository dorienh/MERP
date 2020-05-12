'''
3 types of experiments.

Without profile information:
1) averaged values
2) non averaged values

With profile information:
3) concatenated with profile info

each need a separate dataloader I believe *insert thinking face*
'''

# from abc import ABC, abstractmethod

import os
import sys
sys.path.append(os.path.abspath(''))
print(sys.path)

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import util


##############################################################
####  2) non averaged values without profile information  ####
##############################################################

# this singleSongDataset is for a single song and all labels collected for that song. 
class singleSongDataset(Dataset):

    def __init__(self, song_feat, song_df):
        
        # multiply the features by the number of trials collected for that song
        self.data = list(song_feat)*len(song_df)

        # concatenate the labels of each trial into a list. 
        labels = []
        for df_idx in song_df.index:
            labels.extend(song_df.loc[df_idx, 'labels'])

        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

def form_dataset(feat_dict, exps, labeltype, train=True):
    # remove the column of !labeltype
    exps = exps.drop(columns=[ltype for ltype in util.labeltypes if ltype !=labeltype])
    # rename column to labels
    exps = exps.rename(columns={labeltype:'labels'})

    if train:
        songlist = util.trainlist
    else:
        songlist = util.testlist
    
    # obtain only train/test song features
    sub_feat_dict = {key: feat_dict[key] for key in songlist}

    # obtain only train/test exps
    sub_exps = exps.loc[exps.songurl.str.contains('|'.join(songlist))].reset_index(drop=True)

    dataset_list = []
    for songname, group in sub_exps.groupby('songurl'):
        song_feat = sub_feat_dict[songname]
        songdf = sub_exps.loc[sub_exps.songurl.str.contains(songname)]
        dataset_list.append(singleSongDataset(song_feat, songdf))

    return ConcatDataset(dataset_list)

if __name__ == "__main__":

    # set labeltype here.
    labeltype = 'arousals'

    # set the file paths for features labels and pinfo(if applicable)
    featfile = 'data/feat_dict_ready.pkl'
    labelfile = 'data/exps_ready.pkl'
    
    ## load the data 
    # read audio features from pickle
    feat_dict = util.load_pickle(featfile)
    # read labels from pickle
    exps = pd.read_pickle(labelfile)
    
    dataset = form_dataset(feat_dict, exps, labeltype, train=True)

    loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=0,
        batch_size=32
    )

    for data in loader:
        print(data[0].shape, data[1].shape)
        break
        