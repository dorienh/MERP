'''
3 types of experiments.

Without profile information:
1) averaged values
2) non averaged values

With profile information:
3) concatenated with profile info

each need a separate dataloader I believe *insert thinking face*

also, for time series data
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


# def prep_data(exps, labeltype):
def prep_data(exps, labeltype, train=True):
    # remove the column of !labeltype
    exps = exps.drop(columns=[ltype for ltype in util.labeltypes if ltype !=labeltype])
    # rename column to labels
    exps = exps.rename(columns={labeltype:'labels'})

    if train:
        songlist = util.trainlist
    else:
        songlist = util.testlist
    
    # obtain only train/test song features
    # sub_feat_dict = {key: feat_dict[key] for key in songlist}

    # obtain only train/test exps
    sub_exps = exps.loc[exps.songurl.str.contains('|'.join(songlist))].reset_index(drop=True)

    # return sub_feat_dict, sub_exps
    return sub_exps

##############################################################
####    1) averaged values without profile information    ####
##############################################################



class dataset_ave_no_profile:

    def __init__(self, affect_type, feat_dict, exps):
        self.seed = 42
        self.affect_type = affect_type
        self.feat_dict = feat_dict
        self.ave_exps = self.average_exps_by_songurl(exps)

    def average_exps_by_songurl(self, exps):
        ave_labels = {}
        for songurl, group in exps.groupby('songurl'):
            # ave = np.mean(group['labels'],axis=1)
            # print(f'{songurl} has {len(group)} entries.')
            ave = group['labels'].mean()
            ave_labels[songurl] = ave

            # print(group.head())
        # print(ave_labels)
        return ave_labels

    def gen_dataset(self):

        class averagedDataset(Dataset):

            def __init__(self, feat_dict, ave_exps):
                data = []
                labels = []

                for songurl in feat_dict.keys():
                    data.append(feat_dict[songurl])
                    labels.append(ave_exps[songurl])
                self.data = [item for sublist in data for item in sublist]
                self.labels = [item for sublist in labels for item in sublist]
            

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, index):
                # print(np.shape(self.data[index]),self.labels[index])
                # print(type(self.data[index]), type(self.labels[index]))
                return self.data[index], self.labels[index]

        return averagedDataset(self.feat_dict, self.ave_exps)

##############################################################
####  2) non averaged values without profile information  ####
##############################################################
class dataset_non_ave_no_profile:
        
    def __init__(self, affect_type, feat_dict, exps):
        self.seed = 42
        self.affect_type = affect_type
        self.feat_dict = feat_dict
        # print('meow', feat_dict.keys())
        self.exps = exps

    def gen_dataset(self, train=True):

        # this singleSongDataset is for a single song and all labels collected for that song.
        class singleSongDataset(Dataset):

            def __init__(self, song_feat, song_df):
                
                # multiply the features by the number of trials collected for that song
                self.data = list(song_feat)*len(song_df)
                # print(np.shape(self.data))

                # concatenate the labels of each trial into a list. 
                labels = []
                for df_idx in song_df.index:
                    labels.extend(song_df.loc[df_idx, 'labels'])

                self.labels = labels

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, index):
                # print(np.shape(self.data[index]),self.labels[index])
                # print(type(self.data[index]), type(self.labels[index]))
                return self.data[index], self.labels[index]


        def form_singleSongDataset(train=True):
            # sub_feat_dict, sub_exps = prep_data(self.feat_dict, self.exps, self.affect_type, train=train)
            sub_exps = prep_data(self.exps, self.affect_type, train=train)
            dataset_list = []
            for songname, songdf in sub_exps.groupby('songurl'):
                song_feat = self.feat_dict[songname]
                dataset_list.append(singleSongDataset(song_feat, songdf))

            return ConcatDataset(dataset_list)

        dataset = form_singleSongDataset(train=train)
        # dataset = form_singleSongDataset()
        
        return dataset

##############################################################
####          3) concatenated with profile info           ####
##############################################################
class dataset_non_ave_with_profile:
        
    def __init__(self, affect_type, feat_dict, exps, pinfo_df, conditions):
        self.seed = 42
        self.affect_type = affect_type
        self.feat_dict = feat_dict
        self.exps = exps
        self.conditions = conditions
        desired_columns = ['workerid', *conditions]
        self.pinfo_df = pinfo_df[desired_columns]
    
    def gen_dataset(self, train=True):
        class singlePinfoDataset(Dataset):

            def __init__(self, feat_dict, single_pinfo, p_exps):
                songlist = p_exps.loc[:,'songurl']

                # prep data
                data = []
                for song in songlist:
                    data = [*data, *feat_dict[song]]

                # append profile information to each and every time step of audio features.
                self.data = np.array([list(audiofeat) + list(single_pinfo) for audiofeat in data])
                # print(np.shape(self.data[0]))
                # print(np.shape(self.data))
                # prep labels
                labels = []
                for df_idx in p_exps.index:
                    labels.extend(p_exps.loc[df_idx, 'labels'])

                self.labels = labels
                
            def __len__(self):
                return len(self.data)

            def __getitem__(self, index):
                # print(np.shape(self.data[index]),self.labels[index])
                # print(type(self.data[index]), type(self.labels[index]))
                return self.data[index], self.labels[index]

        def form_singlePinfoDataset(train=train):
            sub_feat_dict, sub_exps = prep_data(self.feat_dict, self.exps, self.affect_type, train=train)
            
            dataset_list = []
            for workerid, p_exps in sub_exps.groupby('workerid'):
                # prep pinfo
                single_pinfo_df = self.pinfo_df.loc[self.pinfo_df.workerid.str.contains(workerid)]

                # cast single_pinfo_df into a list and remove workerid
                single_pinfo = single_pinfo_df.values.tolist()[0][1::]

                dataset_list.append(singlePinfoDataset(sub_feat_dict, single_pinfo, p_exps))

            return ConcatDataset(dataset_list)
        

        dataset = form_singlePinfoDataset(train=train)

        return dataset


if __name__ == "__main__":

    # set labeltype here.
    labeltype = 'arousals'

    train_feat_dict = util.load_pickle('data/train_feats_pca.pkl')
    test_feat_dict = util.load_pickle('data/test_feats_pca.pkl')

    # # set the file paths for features labels and pinfo(if applicable)
    # featfile = 'data/feat_dict_ready.pkl'
    labelfile = 'data/exps_ready.pkl'
    # pinfofile = 'data/pinfo_numero.pkl'
    
    # ## load the data 
    # # read audio features from pickle
    # feat_dict = util.load_pickle(featfile)
    # # read labels from pickle
    exps = pd.read_pickle(labelfile)
    # # read pinfo from pickle
    # pinfo_df = pd.read_pickle(pinfofile)

    sub_exps = prep_data(exps, labeltype, train=True)
    # average_exps_by_songurl(sub_exps)

    dataset_obj = dataset_ave_no_profile(labeltype, train_feat_dict, sub_exps)
    dataset = dataset_obj.gen_dataset()
    # dataset_obj = dataset_non_ave_with_profile(labeltype, feat_dict, exps, pinfo_df, ['age'])
    # dataset = dataset_obj.gen_dataset(False)

    # dataset_obj = dataset_non_ave_no_profile(labeltype, train_feat_dict, exps)
    # dataset = dataset_obj.gen_dataset()

    loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=0,
        batch_size=32
    )

    # # torch.save(loader, 'method/pinfo_dataloader.pth')

    for data in loader:
        print(np.shape(data[0]), np.shape(data[1]))
        # print(data)
        break
    # '''
    # averaged_arousals = {}
    # for songurl, song_group in exps.groupby('songurl'):
    #     arousals_np = song_group['arousals'].to_numpy()
    #     ave_arousals = np.average(arousals_np, axis=0)
    #     averaged_arousals[songurl] = ave_arousals
    
    # def check_dict_keys_and_shape(dict_obj):
    #     for k,v in dict_obj.items():
    #         print(f'key: {k} || item_len: {len(v)}')
    
    # check_dict_keys_and_shape(averaged_arousals)
    # '''