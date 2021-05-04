
import os
import sys
sys.path.append(os.path.abspath(''))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import util





##############################################################
####    1) averaged values without profile information    ####
##############################################################

class dataset_ave_no_profile:

    def __init__(self, feat_dict, exps, train):
        '''
            feat_dict - the audio feature dictionary
            exps - should be the dataframe read from exps_std_a/v_ave.pkl
            train - boolean
        '''
        # self.seed = 42
        self.feat_dict = feat_dict
        # self.ave_exps = self.exps # expects exps_std_a/v_ave.pkl
        # if train:
        #     songlist = util.trainlist
        # else:
        #     songlist = util.testlist
        songlist = self.feat_dict.keys()
        self.ave_exps = exps.loc[exps.index.str.contains('|'.join(songlist))]
        # print(self.ave_exps.head())
        # self.ave_exps = self.ave_exps.set_index('songurl')

    def gen_dataset(self):

        class averagedDataset(Dataset):

            def __init__(self, feat_dict, ave_exps):
                data = []
                labels = []

                for songurl in feat_dict.keys():
                    # print(songurl)
                    data.append(feat_dict[songurl])
                    labels.append(ave_exps.at[songurl, 'labels'])
                    # print(labels[-1][0:5][0:5])
                self.data = [item for sublist in data for item in sublist]
                self.labels = [item for sublist in labels for item in sublist]
            

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, index):
                # print(np.shape(self.data[index]),self.labels[index])
                # print(type(self.data[index]), type(self.labels[index]))
                return self.data[index], self.labels[index]

        return averagedDataset(self.feat_dict, self.ave_exps)

class rdm_dataset(Dataset):

    def __init__(self, feat_dict, ave_exps, seq_len=1, seed=42):

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

##############################################################
####                    MAIN FUNCTION                     ####
##############################################################

if __name__ == "__main__":

    labeltype = 'arousals'

    # load the data 
    # read audio features from pickle
    # train_feat_dict = util.load_pickle('data/train_feats_pca.pkl')
    # test_feat_dict = util.load_pickle('data/test_feats_pca.pkl')

    # train_feat_dict = util.load_pickle('data/folds/train_feats_pca_1.pkl')
    # test_feat_dict = util.load_pickle('data/folds/test_feats_pca_1.pkl')
    train_feat_dict = util.load_pickle('data/train_feats.pkl')
    test_feat_dict = util.load_pickle('data/test_feats.pkl')

    # read labels from pickle
    exps = pd.read_pickle('data/exps_std_a_ave3.pkl')
    # exps = exps.set_index('songurl')
    # print(exps.head())
    # print(exps.at['00_145', 'labels'])
    # print(exps.index)

    # dataset_obj = dataset_ave_no_profile(train_feat_dict, exps, train=True)
    # dataset = dataset_obj.gen_dataset()

    dataset = rdm_dataset(test_feat_dict, exps)

    loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=0,
        batch_size=8
    )

    for data in loader:
        print(data[0], data[1])
        break

    # print(np.shape(list(train_feat_dict.values())[0])[1])