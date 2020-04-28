import os
import pandas as pd
import numpy as np
import glob
import torch
import pickle

featurepaths = glob.glob(os.path.join(os.path.abspath(''), 'data', '500ms_100hop', '*.csv'))

def load_features_from_csv(feature_paths):
    feat_dict = {}
    numpaths = len(feature_paths)
    for idx, fpath in enumerate(feature_paths):
        songurl = os.path.basename(fpath)[:-4]
        print('processing song {} / {} : {} '.format(idx+1, numpaths, songurl))
        feat_dict[songurl] = pd.read_csv(fpath).to_numpy()[:, 1:]
    return feat_dict

def main():

    feat_dict = load_features_from_csv(featurepaths)
    # print(feat_dict.keys())
    with open('data/feat_dict.pkl', 'wb') as handle:
        pickle.dump(feat_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()