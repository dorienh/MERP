'''
window and partitian the features and labels. leave the labels unaveraged? hmm..
'''

# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


exps = pd.read_pickle(os.path.join(os.path.abspath('..'), 'data', 'exps_ready2.pkl'))

# %%
exps.shape
# %%
exps.workerid.unique().shape

# %%
