'''
1) Remove the fist 15 seconds of audio features and participant given labels
2) Resample to 0.5 seconds pre label and per feature.
'''

#%%
import os
import sys
sys.path.append(os.path.abspath(''))
print(sys.path)
import pickle
import numpy as np
import pandas as pd