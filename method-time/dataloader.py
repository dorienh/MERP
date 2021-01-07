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
# print(sys.path)

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import util


def prep_data(feat_dict, exps, labeltype, train=True):
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

    return sub_feat_dict, sub_exps

##############################################################
####    1) averaged values without profile information    ####
##############################################################




##############################################################
####  2) non averaged values without profile information  ####
##############################################################
class dataset_non_ave_no_profile:
        
    def __init__(self, affect_type, feat_dict, exps, lstm_size, step):
        self.seed = 42
        self.affect_type = affect_type
        self.feat_dict = feat_dict
        self.exps = exps
        self.lstm_size = lstm_size
        self.step = step

    def gen_dataset(self, train=True):

        # this singleSongDataset is for a single song and all labels collected for that song.
        class singleSongDataset(Dataset):

            def __init__(self, song_feat, song_df, lstm_size, step):
                
                # multiply the features by the number of trials collected for that song
                # audiofeat = list(song_feat)*len(song_df)
                audiofeat = np.array(song_feat)
                audiofeat = torch.from_numpy(audiofeat)
                print('meow')
                print(audiofeat.shape)
                w_audiofeat = audiofeat.unfold(0,lstm_size, step)
                print(len(song_df), np.shape(w_audiofeat))
                reverse = w_audiofeat.fold
                self.data = torch.cat(len(song_df)*[w_audiofeat])
                print(np.shape(self.data))

                # concatenate the labels of each trial into a list. 
                labels = []
                for df_idx in song_df.index:
                    label = song_df.loc[df_idx, 'labels']
                    label = torch.from_numpy(label)
                    w_label = label.unfold(0,lstm_size, step)
                    # print(np.shape(w_label))
                    labels.extend(w_label)

                self.labels = labels
                # print(len(labels))

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, index):
                # print(np.shape(self.data[index]),self.labels[index])
                # print(type(self.data[index]), type(self.labels[index]))
                return self.data[index], self.labels[index]

        def form_singleSongDataset(train=True):
            sub_feat_dict, sub_exps = prep_data(self.feat_dict, self.exps, self.affect_type, train=train)

            dataset_list = []
            for songname, songdf in sub_exps.groupby('songurl'):
                song_feat = sub_feat_dict[songname]
                dataset_list.append(singleSongDataset(song_feat, songdf, self.lstm_size, self.step))
                break

            return ConcatDataset(dataset_list)

        dataset = form_singleSongDataset(train=train)
        
        return dataset

##############################################################
####          3) concatenated with profile info           ####
##############################################################
class dataset_non_ave_with_profile:
        
    def __init__(self, affect_type, feat_dict, exps, pinfo_df, conditions, lstm_size, step):
        self.seed = 42
        self.affect_type = affect_type
        self.feat_dict = feat_dict
        self.exps = exps
        self.conditions = conditions
        # print('meow  ', conditions)
        desired_columns = ['workerid', *conditions]
        self.pinfo_df = pinfo_df[desired_columns]
        self.lstm_size = lstm_size
        self.step = step
    
    def gen_dataset(self, train=True):
        class singlePinfoDataset(Dataset):

            def __init__(self, feat_dict, single_pinfo, p_exps, lstm_size, step):
                songlist = p_exps.loc[:,'songurl']
                pinfo = torch.from_numpy(np.array(single_pinfo))
                # print("pinfo: ", pinfo)

                # prep data
                data = []
                for song in songlist:
                    # window for input to lstm
                    audiofeat = np.array(feat_dict[song])
                    audiofeat = torch.from_numpy(audiofeat)
                    w_audiofeat = audiofeat.unfold(0,lstm_size, step)
                    # duplicate pinfo like mad 
                    repeated_pinfo = pinfo.repeat(len(w_audiofeat),lstm_size,1).permute(0,2,1)
                    data.append(torch.cat((w_audiofeat, repeated_pinfo),dim=1))
                
                # append profile information to each and every time step of audio features.
                self.data = torch.cat(data)
                # print(self.data.shape)

                # prep labels
                labels = []
                for df_idx in p_exps.index:
                    label = p_exps.loc[df_idx, 'labels']
                    label = torch.from_numpy(label)
                    w_label = label.unfold(0,lstm_size, step)
                    # print(np.shape(w_label))
                    labels.extend(w_label)

                self.labels = labels

                # print('label: ', labels)
                
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
                # print('numsongs: ', len(p_exps))
                # prep pinfo
                single_pinfo_df = self.pinfo_df.loc[self.pinfo_df.workerid.str.contains(workerid)]
                # single_pinfo_df = single_pinfo_df[self.conditions]

                # cast single_pinfo_df into a list and remove workerid
                single_pinfo = single_pinfo_df.values.tolist()[0][1::]

                dataset_list.append(singlePinfoDataset(sub_feat_dict, single_pinfo, p_exps, self.lstm_size, self.step))

            return ConcatDataset(dataset_list)
        

        dataset = form_singlePinfoDataset(train=train)

        return dataset


if __name__ == "__main__":

    # set labeltype here.
    labeltype = 'arousals'
    lstm_size = 10
    step = 10
    conditions = ['age']

    # set the file paths for features labels and pinfo(if applicable)
    featfile = 'data/feat_dict_ready.pkl'
    labelfile = 'data/exps_ready.pkl'
    pinfofile = 'data/pinfo_numero.pkl'
    
    # load the data 
    # read audio features from pickle
    feat_dict = util.load_pickle(featfile)
    # read labels from pickle
    exps = pd.read_pickle(labelfile)
    # read pinfo from pickle
    pinfo_df = pd.read_pickle(pinfofile)

    # for wid, group in exps.groupby('workerid'):
    #     print(len(group))

    # dataset_obj = dataset_non_ave_with_profile(labeltype, feat_dict, exps, pinfo_df, conditions, lstm_size, step)
    # dataset = dataset_obj.gen_dataset(False)

    dataset_obj = dataset_non_ave_no_profile(labeltype, feat_dict, exps, lstm_size, step)
    dataset = dataset_obj.gen_dataset(False)

    loader = DataLoader(
        dataset,
        shuffle=True,
        num_workers=0,
        batch_size=32
    )

    # # torch.save(loader, 'method/pinfo_dataloader.pth')

    # for data in loader:
    #     print(np.shape(data[0]), np.shape(data[1]))
    #     # print(data)
    #     break
    '''
    averaged_arousals = {}
    for songurl, song_group in exps.groupby('songurl'):
        arousals_np = song_group['arousals'].to_numpy()
        ave_arousals = np.average(arousals_np, axis=0)
        averaged_arousals[songurl] = ave_arousals
    
    def check_dict_keys_and_shape(dict_obj):
        for k,v in dict_obj.items():
            print(f'key: {k} || item_len: {len(v)}')
    
    check_dict_keys_and_shape(averaged_arousals)
    '''