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
# affect_type = 'arousals'
exps = pd.read_pickle('../../data/exps_ready.pkl')


# %%
for songurl, group in exps.groupby('songurl'):
    arousal_series = group['arousals'].to_numpy()
    # valence_series = group['valences'].to_numpy()
    worker_series = group['workerid'].to_numpy()
    # temp = [print(worker) for label,worker in zip(label_series, worker_series) ]
    song_df = pd.DataFrame.from_records(arousal_series)

    fig = plt.figure(figsize=(20,10))

    sns.heatmap(song_df, vmin=-1, vmax=1)
    plt.savefig(f'../plots/heatmaps/{songurl}_partcipant_time.png')
    

# %%
exp_log_filepath = os.path.join(dir_path,'saved_models','experiment_log2.pkl')
pd.read_pickle(exp_log_filepath)
