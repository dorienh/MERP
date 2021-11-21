'''
profile information is concatenated to the audio features at every time step
'''

import os
import sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))

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

        songlist = feat_dict.keys()
        ave_exps = ave_exps.loc[ave_exps.songurl.str.contains('|'.join(songlist))]
        ave_exps = ave_exps.reset_index(drop=True)

        self.feat_dict = feat_dict
        self.exps = ave_exps

    def __getitem__(self,index):

        row = self.exps.iloc[index]
        # profile
        profile = row['profile']
        if hasattr(profile, '__iter__'):
            profile = list(profile)
        else:
            profile = [profile]

        # audio 
        songurl = row['songurl']
        audio_feat_full = self.feat_dict[songurl]

        # label
        label_full = row['labels']

        if self.seq_len:
            audio_length = len(audio_feat_full)
            # print(np.shape(audio_feat_full))
            start_idx = self.random.randint(audio_length - self.seq_len)
            end_idx = start_idx + self.seq_len

            audio_feat = audio_feat_full[start_idx:end_idx]
            label = label_full[start_idx:end_idx]

        else:
            audio_feat = audio_feat_full
            label = label_full

        # concatenate audio feature and profile features (duplicated for every time step)
        # print(np.shape(audio_feat))
        repeated_profile = [profile for a in np.arange(np.shape(audio_feat)[0])]
        # print('nya', np.shape(repeated_profile))
        data = np.concatenate((audio_feat, repeated_profile),axis=1)
        # print(np.shape(data))

        
       

        return data, label
    
    def __len__(self):
        return len(self.exps)

'''
class linear_dataset(Dataset):

    def __init__(self, feat_dict, ave_exps):

        songlist = feat_dict.keys()

        ave_exps = ave_exps.loc[ave_exps.songurl.str.contains('|'.join(songlist))]
        ave_exps = ave_exps.reset_index(drop=True)
        
        # count the total number of individual labels
        labels = np.concatenate(ave_exps['labels'].to_list())
        print(len(labels))


        self.feat_dict = feat_dict
        self.exps = ave_exps

    def __getitem__(self,index):

        row = self.exps.iloc[index]
        
        # audio 
        songurl = row['songurl']
        audio_feat_full = self.feat_dict[songurl]

        audio_length = len(audio_feat_full)
        # print(np.shape(audio_feat_full))
        start_idx = self.random.randint(audio_length - self.seq_len)
        end_idx = start_idx + self.seq_len

        audio_feat = audio_feat_full[start_idx:end_idx]

        # profile
        profile = list(row['profile'])

        # concatenate audio feature and profile features (duplicated for every time step)
        # print(np.shape(audio_feat))
        repeated_profile = [profile for a in np.arange(np.shape(audio_feat)[0])]
        # print('nya', np.shape(repeated_profile))
        data = np.concatenate((audio_feat, repeated_profile),axis=1)
        # print(np.shape(data))

        # label
        label_full = row['labels']
        label = label_full[start_idx:end_idx]

        return data, label
    
    def __len__(self):
        return len(self.exps)
'''

if __name__ == "__main__":

    affect_type = 'arousals'

    train_feat_dict = util.load_pickle('data/train_feats.pkl')
    # test_feat_dict = util.load_pickle('../data/test_feats.pkl')
    exps = pd.read_pickle(os.path.join('data', f'exps_std_{affect_type[0]}_profile_ave_1.pkl'))

    dataset = rdm_dataset(train_feat_dict, exps, seq_len=None)
    loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=0,
        batch_size=1
    )

    for j in np.arange(10):
        for i, data in enumerate(loader):
            # print(np.shape(data[0]), np.shape(data[1]))
            print(data[0][0][0][0:10])
            
            if i>2:
                break
