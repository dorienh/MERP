
import os
import sys
sys.path.append(os.path.abspath(''))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import util

class rdm_dataset(Dataset):

    def __init__(self, feat_dict, ave_exps, seq_len=10, seed=42):

        self.random = np.random.RandomState(seed)
        self.seq_len = seq_len

        self.song_lengths = [len(a) for a in feat_dict.values()]
        
        self.data = []
        self.labels = []

        # extract labels only in the songlist.
        songlist = feat_dict.keys()
        self.num_songs = len(songlist)
        ave_exps = ave_exps.loc[ave_exps.index.str.contains('|'.join(songlist))]

        for songurl in feat_dict.keys():
            self.data.append(feat_dict[songurl])
            self.labels.append(ave_exps.at[songurl, 'labels'])
        
        # self.data = [item for sublist in data for item in sublist]
        # self.labels = [item for sublist in labels for item in sublist]

    def __getitem__(self,index):

        data = self.data[index]
        label = self.labels[index]

        audio_length = self.song_lengths[index]
        start_idx = self.random.randint(audio_length - self.seq_len)
        end_idx = start_idx + self.seq_len

        return data[start_idx:end_idx], label[start_idx:end_idx]
    
    def __len__(self):
        return len(self.data)

    '''
    def __getitem__(self, index):
        idx = index % self.num_songs
        data = self.data[idx]
        label = self.labels[idx]
        audio_length = self.song_lengths[idx]
        start_idx = self.random.randint(audio_length - self.seq_len)
        end_idx = start_idx + self.seq_len
        if len(data[start_idx:end_idx]) <10:
            print(len(data[start_idx:end_idx]), len(label[start_idx:end_idx]))

        return data[start_idx:end_idx], label[start_idx:end_idx]
        


    def __len__(self):
        return self.num_songs*100
    '''
