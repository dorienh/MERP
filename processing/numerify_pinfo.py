# from matplotlib.colors import get_named_colors_mapping
import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(''))
# sys.path.append(os.path.abspath('../..'))
print(sys.path)
# import util

mapper_dict = {
    'age': {(0,25):0.0, (26,35):0.33, (36,50):0.66, (51,80):1.0},
    'gender': {'Female': 1.0, 'Other': 0.5, 'Male': 0.0},
    'residence': {'US': 1.0, 'IN': 0.5, 'Other': 0.0},
    'enculturation': {'US': 1.0, 'IN': 0.5, 'Other': 0.0},
    'language': {'EN': 1.0, 'TA': 0.5, 'Other': 0.0},
    'genre': {'Rock': 1.0, 'Classical music': 0.66, 'Pop': 0.33, 'Other': 0.0},
    'instrument': {'Yes': 1.0, 'No': 0.0},
    'training': {'Yes': 1.0, 'No': 0.0},
    'duration': {(0,0):0, (1,5):0.5, (6,50):1.0},
    'master': {'Yes': 1.0, 'No': 0.0}
}

# reversed in va_by_profile.py
# not sure if i'll use it but might be useful for plotting output. 
r_mapper_dict = {'age': {0.0: (0, 25), 0.33: (26, 35), 0.66: (36, 50), 1.0: (51, 80)}, 
'gender': {1.0: 'Female', 0.5: 'Other', 0.0: 'Male'}, 
'residence': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'enculturation': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'language': {1.0: 'English', 0.5: 'Tamil', 0.0: 'Other'}, 
'genre': {1.0: 'Rock', 0.66: 'Classical', 0.33: 'Pop', 0.0: 'Other'}, 
'instrument': {1.0: 'Yes', 0.0: 'No'}, 
'training': {1.0: 'Yes', 0.0: 'No'}, 
'duration': {0: (0, 0), 0.5: (1, 5), 1.0: (6, 50)}, 
'master': {1.0: 'Yes', 0.0: 'No'}}

def categorical(mapper, arr):
    retval = list(map(mapper.get, arr))
    retval = [a if a is not None else 0 for a in retval] ## handling "Others"
    return retval

def numerical(range_map, arr):
    def get_cat(range_map, x):
        for key in range_map:
            x = int(x)
            if key[0] <= x <= key[1]:
                return range_map[key]
    retval = [get_cat(range_map, a) for a in arr]
    # for a in arr:
        # print(a, ' || ', get_cat(range_map, a))
    return retval

# load pinfo
pinfo = pd.read_pickle('data/mediumrare/semipruned_pinfo.pkl')
exps = pd.read_pickle('data/exps_ready3.pkl')
pinfo = pinfo[pinfo['workerid'].isin(exps['workerid'].unique())]
# print(pinfo)

n_pinfo_dict = {
    'workerid': pinfo['workerid']
}

for profile_type, mapper in mapper_dict.items():
    col = pinfo[profile_type].to_numpy()
    if profile_type in ['age', 'duration']:
        ncol = numerical(mapper, col)
    else:
        ncol = categorical(mapper, col)
    # print(profile_type, pd.Series(ncol).value_counts())
    n_pinfo_dict[profile_type] = ncol

df = pd.DataFrame(n_pinfo_dict)
print(df.head())
print(df.shape)

df.to_pickle(os.path.join(os.path.abspath(''), 'data', 'pinfo_numero.pkl'))

