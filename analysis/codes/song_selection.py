#%%
'''
imports
'''
import os
import glob
import pandas as pd
import numpy as np

# %%
filepath = '/home/meowyan/Documents/emotion_paper/song_picking/*.csv'
files = glob.glob(filepath)

# %%
temp = pd.read_csv(files[0])

# %%
