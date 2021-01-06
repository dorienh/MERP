#%%
import os
import sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))

import pandas as pd
import util
# %%
exp_log_filepath = os.path.join('saved_models','kfold','experiment_log.pkl')
util.load_pickle(exp_log_filepath)
# %%
