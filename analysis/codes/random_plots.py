# %%

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


exps = pd.read_pickle(os.path.join(os.path.abspath('../..'), 'data', 'exps_ready.pkl'))


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


