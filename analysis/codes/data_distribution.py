# %%
# heat map??
import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
print(sys.path)
import util
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# %%
affect_type = 'valences'
exps = pd.read_pickle('../../data/exps_ready.pkl')


# %%
for songurl, group in exps.groupby('songurl'):
    arousal_series = group[affect_type].to_numpy()
    # valence_series = group['valences'].to_numpy()
    worker_series = group['workerid'].to_numpy()
    # temp = [print(worker) for label,worker in zip(label_series, worker_series) ]
    song_df = pd.DataFrame.from_records(arousal_series)

    fig = plt.figure(figsize=(20,10))

    sns.heatmap(song_df, vmin=-1, vmax=1)
    plt.savefig(f'../plots/heatmaps/{affect_type}/{songurl}_partcipant_time.png')
    
    

# %%
exp_log_filepath = os.path.join(dir_path,'saved_models','experiment_log2.pkl')
pd.read_pickle(exp_log_filepath)


#%%
# EXTRA - find the total length of the 54 songs put together. min, max and mean lengths too.
import os
import sys
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
import util

import librosa

songfilenamedict = {}

filepath = '/home/meowyan/Documents/emotion/data/50songs'
for e in util.songlist:
    one, two = e.split('_')
    song_filepath = os.path.join(filepath, one, f'{two}.wav')
    songfilenamedict[e] = song_filepath
    # this is inefficient but... i get to see the time of each song so maybe it's worth it lol
songlengthdict = {}
for key, path in songfilenamedict.items():
    y, sr = librosa.load(path)
    length = round(len(y)/ sr *10, 2)
    print(length)
    songlengthdict[key] = length
    



# %%
import numpy as np

lengths = np.array(list(songlengthdict.values()))
print('min length: ', min(lengths))
print('max length: ', max(lengths))
print('mean length: ', np.mean(lengths))
print('sum length: ', sum(lengths))
'''
00 sum: 20462.82 (34.1 mins)
01 sum: 13775.09 (22.96 mins)
0505 sum: 18870.58 (31.45 mins)
10 sum: 9622.18 (16.04 mins)
11 sum: 23245.159 (38.74 mins)
'''
# %%
