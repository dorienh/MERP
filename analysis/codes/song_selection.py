#%%
'''
imports
'''
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %%
filepath = '/home/meowyan/Documents/emotion_paper/song_picking/FMA_av_prediction.csv'
# files = glob.glob(filepath)

# %%
temp = pd.read_csv(filepath)

# %%
pinfo = pd.read_pickle(os.path.join('../../data', 'pinfo_numero.pkl'))
print(pinfo)

# %%
