# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


exps = pd.read_pickle(os.path.join(os.path.abspath('../..'), 'data', 'exps_ready3.pkl'))


# %%
'''
number of participants who labelled each song
'''

count = exps.groupby('songurl').size()



# %%
# discard the pie plot idea, way too messy. 
# count.plot.pie()
plt.figure(figsize=(14,5))
count[:-4].plot.bar()
plt.xlabel('song')
plt.ylabel('number of participants')
plt.savefig('../plots/num_p_per_song.png')
plt.close()
'''

deam songs not plotted because all participants labelled them. all 197 of them. 
'''
# %%

import os
import sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..')
)
import util_method

import numpy as np


temp = np.arange(351)

a = util_method.windowing(temp, 10, 10)
print(len(a))
print(a[0])
print(len(a[-1]))
b = util_method.reverse_windowing(a, 10, 10)
print(len(b))
print(b[0:20])
print(b[-20:])


# %%
