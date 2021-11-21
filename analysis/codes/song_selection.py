#%%
'''
imports
'''
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 300
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
import seaborn as sns

# %%
#FMA
filepath = '/home/meowyan/Documents/emotion_paper/song_picking/FMA_av_prediction.csv'
fma_av = pd.read_csv(filepath)

#%%
def binning(y):
    y_list = []
    boundary1=0.31
    boundary2=0.45
    boundary3=0.56
    boundary4=0.67
    for i in y:
        if i < boundary1:
            y_list.append(0)
        elif np.logical_and(i >= boundary1, i < boundary2):
            y_list.append(1)
        elif np.logical_and(i >= boundary2, i < boundary3):
            y_list.append(2)
        elif np.logical_and(i >= boundary3, i < boundary4):
            y_list.append(3)
        else:
            y_list.append(4)
            
    return y_list

fma_av['arousal_bins'] = binning(fma_av['arousal'])
fma_av['valence_bins'] = binning(fma_av['valence'])

'''
FMA is a large database that consists of 106,574 tracks. 
From the list of top 1000 songs listened to, we filtered out the songs shorter that 30 seconds and longer than 10 minutes. To the remaining songs we then determined a static arousal value and valence value for each. The arousal valence values of the songs are shown in Fig{bla}. 
'''

#%%
fma_av['arousal'] = fma_av['arousal']*2-1
fma_av['valence'] = fma_av['valence']*2-1

#%%

fig = plt.figure()


ax = fig.add_subplot(111)


sns.scatterplot(x='arousal', y='valence', data=fma_av)

ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')

# sns.scatterplot(x='')


#%%
fma_av[fma_av['songID']=='Chart_33']
group1 = ['Chart_33','Chart_58','Chart_80','Chart_62','Chart_90','Chart_184','Chart_46','Chart_85','Chart_102','Chart_199']
group2 = ["Chart_209", "Chart_411", "Chart_427", "Chart_459", "Chart_505", "Chart_528", "Chart_648", "Chart_693", "Chart_748", "Chart_801"]
group3 = ['Chart_71', 'Chart_139', 'Chart_143', 'Chart_154', 'Chart_175', 'Chart_177', 'Chart_228', 'Chart_333', 'Chart_890', 'Chart_897']
group4 = ['Chart_93', 'Chart_130', 'Chart_150', 'Chart_216', 'Chart_243', 'Chart_288', 'Chart_404', 'Chart_487', 'Chart_828', 'Chart_942']
group5 = ['Chart_35', 'Chart_145', 'Chart_275', 'Chart_366', 'Chart_661', 'Chart_695', 'Chart_702', 'Chart_839', 'Chart_882', 'Chart_883']

fma_av['group'] = 'not selected'
print(fma_av)
#%%
for i, row in fma_av.iterrows():
    if row['songID'] in group1:
        fma_av.loc[i, 'group'] = 'v:05 a:05'
    if row['songID'] in group2:
        fma_av.loc[i, 'group'] = 'v:1  a:1'
    if row['songID'] in group3:
        fma_av.loc[i, 'group'] = 'v:1  a:0'
    if row['songID'] in group4:
        fma_av.loc[i, 'group'] = 'v:0  a:1'
    if row['songID'] in group5:
        fma_av.loc[i, 'group'] = 'v:0  a:0'

print(fma_av.loc[30])
fma_av = fma_av.sort_values(by='group')
#%%

fig = plt.figure(figsize=(10,10))
plt.xlim(-1,1)
plt.ylim(-1,1)


ax = fig.add_subplot(111)

# palette = sns.color_palette('bright', 5)
grey = (0.8117647058823529, 0.8117647058823529, 0.8117647058823529)
red = (0.9098039215686274, 0.0, 0.043137254901960784)
purple = (0.5450980392156862, 0.16862745098039217, 0.8862745098039215)
brown = (0.6235294117647059, 0.2823529411764706, 0.0)
pink = (0.9450980392156862, 0.2980392156862745, 0.7568627450980392)
blue = (0.0, 0.8431372549019608, 1.0)
palette = [grey, red, purple, brown, pink, blue]
sns.scatterplot(x='arousal', y='valence', palette=palette, hue='group', data=fma_av)

ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')


ax.set_xticks([-1,1])
ax.set_yticks([-1,1])

ax.set_ylabel('valence',labelpad=270,size=18)
ax.set_xlabel('arousal',labelpad=270,size=18)

#%%
from PIL import Image
from io import BytesIO

png1 = BytesIO()
fig.savefig(png1, format='png')

# (2) load this image into PIL
png2 = Image.open(png1)

# (3) save as TIFF
png2.save('../plots/selected_fma.tif')
png2.save('../plots/selected_fma.png')
png1.close()

#%%
fig = plt.figure(figsize=(10,10))
plt.xlim(-1,1)
plt.ylim(-1,1)

ax = fig.add_subplot(111)
palette = [grey, red, purple, brown, pink, blue]
sns.scatterplot(x='arousal', y='valence', data=fma_av)

ax.spines['left'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_color('none')

ax.set_xticks([-1,1])
ax.set_yticks([-1,1])

ax.set_ylabel('valence',labelpad=270,size=18)
ax.set_xlabel('arousal',labelpad=270,size=18)

# %%
pinfo = pd.read_pickle(os.path.join('../../data', 'pinfo_numero.pkl'))
print(pinfo)

# %%
exp_log = pd.read_pickle('/home/meowyan/Documents/emotion/method-2networks/saved_models/experiment_log4.pkl')
exp_log.to_csv('/home/meowyan/Documents/emotion/method-2networks/saved_models/experiment_log4.csv')

# %%

def plot_scatter(y_pred, t_X2d, num_clusters):
    y_binned = binning(y_pred)
    palette5 = sns.color_palette('bright',num_clusters)
    sns.scatterplot(t_X2d[:,0],t_X2d[:,1],palette=palette5, hue=y_binned, legend='full')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
