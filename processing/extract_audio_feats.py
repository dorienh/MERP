import os
import pandas as pd
import numpy as np
import glob
import torch
import pickle

# featurepaths = glob.glob(os.path.join(os.path.abspath(''), 'data', '500ms_100hop', '*.csv'))
# featurepaths = glob.glob(os.path.join(os.path.abspath('..'), 'data', 'songs_csvs_ComParE', '*.csv'))
featurepaths = glob.glob('/home/meowyan/Documents/emotion/data/song_csvs_ComParE/*.csv')
# print(featurepaths)

def load_features_from_csv(feature_paths):
    feat_dict = {}
    numpaths = len(feature_paths)
    for idx, fpath in enumerate(feature_paths):
        songurl = os.path.basename(fpath)[:-4]
        # print(songurl)
        print('processing song {} / {} : {} '.format(idx+1, numpaths, songurl))
        feat_dict[songurl] = pd.read_csv(fpath, sep=';').to_numpy()[:, 1:]

        # print(pd.read_csv(fpath).to_numpy().shape)
        # temp = pd.read_csv(fpath,sep=';').to_numpy()[1:, :]
        # print(type(temp[1]))
        # print(temp[0].shape)
        # print(temp[0:2])
        # break
        print(f'len: {len(feat_dict[songurl])}')
    return feat_dict

def main():

    feat_dict = load_features_from_csv(featurepaths)
    # print(feat_dict.keys())
    with open('data/feat_dict2.pkl', 'wb') as handle:
        pickle.dump(feat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # print(feat_dict['deam_115'])

if __name__ == '__main__':
    main()