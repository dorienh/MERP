'''
I ran pruning.py until exp3 before doing deam comparison. 
'''

#%%
'''
imports
'''
import os
import glob
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
print(sys.path)
import matplotlib.pyplot as plt

import util

#%%
'''
read deam annotations for the 4 deam songs we used
10 annotations per song exist.
'''
datatype = 'arousals'
deampath = '../data/deam_annotations/annotations_per_each_rater/dynamic-persec'
deamsonglist = ['115', '343', '745', '1334']
# %%

deamlabels = {}
for deamsong in deamsonglist:
    path = os.path.join(deampath, datatype[:-1], f'{deamsong}.csv')
    print(path)
    deamlabels[f'deam_{deamsong}'] =  pd.read_csv(path, index_col=None, header=0) 

# %%
'''
plot the deam song average and standard deviation
'''

def plot_deam_ave_std():
    for songurl, df in deamlabels.items():
        ave = df.mean(axis=0)
        stddev = df.std(axis=0)
        fig = plt.figure(figsize=(14,6))
        plt.plot(ave, label='ave')
        plt.plot(ave+stddev, label='ave+std')
        plt.plot(ave-stddev, label='ave-std')
        plt.ylim(-1.1,1.1)
        plt.legend()
        plt.xticks(ticks=np.arange(0,60,2), labels=np.arange(15,45,1))
        plt.xlabel('time')
        plt.ylabel(f'{datatype[:-1]}')
        plt.title(f'Average of {datatype} for {songurl} across 10 participants')
        # plt.show()
        plt.savefig(f'../analysis/plots/deam_ave_std_persong/{datatype[:-1]}/{songurl}.png')
        plt.close()

plot_deam_ave_std()
# %%
def get_deamstats():
    deamstats = {}
    for songurl, df in deamlabels.items():
        ave = df.mean(axis=0)
        stddev = df.std(axis=0)
        deamstats[songurl] = {'ave':ave.to_numpy(), 'stddev':stddev.to_numpy()}
    return deamstats

deamstats = get_deamstats()

#%%
def average_1D(arr, n):
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)

def check_if_within_std(deamstat, ourlabel, std_mult=1):
    maxlist = deamstat['ave'] + deamstat['stddev']*std_mult
    minlist = deamstat['ave'] - deamstat['stddev']*std_mult

    within_bool_list = []

    for mindeam, ours, maxdeam in zip(minlist, ourlabel, maxlist):
        if mindeam < ours < maxdeam:
            within_bool_list.append(True)
        else:
            within_bool_list.append(False)
    return within_bool_list




# %%
'''
print ourlabel over deam label
'''
def plot_collected_against_deam_each_participant(songurl, ourlabel, percentage_match):
    ave = deamstats[songurl]['ave']
    stddev = deamstats[songurl]['stddev']
    fig = plt.figure(figsize=(14,6))
    plt.plot(ave, label='ave')
    plt.plot(ave+stddev, label='ave+std')
    plt.plot(ave-stddev, label='ave-std')
    plt.plot(ourlabel, label='ours')
    plt.ylim(-1.1,1.1)
    plt.legend()
    plt.xticks(ticks=np.arange(0,60,2), labels=np.arange(15,45,1))
    plt.xlabel('time')
    plt.ylabel(f'{datatype[:-1]}')
    plt.title(f'Comparing average of {datatype} for {songurl} between DEAM and COLLECTED dataset. Percentage of match = {percentage_match}%')
    
    return plt

def plot_collected_against_deam_all_participants(songurl, ourlabels, ave_percentage_match):
    ave = deamstats[songurl]['ave']
    stddev = deamstats[songurl]['stddev']
    fig = plt.figure(figsize=(14,6))
    plt.plot(ave, label='ave')
    plt.plot(ave+stddev, label='ave+std')
    plt.plot(ave-stddev, label='ave-std')
    plt.plot(ourlabels[0], 'r', label='ours')
    for ourlabel in ourlabels[1::]:
        plt.plot(ourlabel, 'r')
    plt.ylim(-1.1,1.1)
    plt.legend()
    plt.xticks(ticks=np.arange(0,60,2), labels=np.arange(15,45,1))
    plt.xlabel('time')
    plt.ylabel(f'{datatype[:-1]}')
    plt.title(f'Comparing average of {datatype} for {songurl} between DEAM and COLLECTED dataset. Average percentage of match = {ave_percentage_match}%')
    return plt


# %%
'''
per participant
'''

deamtrials = exps3[exps3['songurl'].str.contains('deam')]
percentage_match_list = []
percentmatchdict = {}
for idx in deamtrials.index:
    print(f'trial number {idx}')
    trial = exps3.iloc[idx]
    songurl = trial['songurl']
    # rescale to 0.5 and cut off 15 seconds in the front.
    ourlabel = average_1D(trial[datatype],5)[30::]
    temp = check_if_within_std(deamstats[songurl], ourlabel)
    percentage_match = round((sum(temp)/len(temp))*100, 2)
    percentage_match_list.append(percentage_match)
    # I want to be able to determine which participant to remove from the dataset but what's the threshold?
    # percentmatchdict[]


    plt = plot_collected_against_deam_each_participant(songurl,ourlabel,percentage_match)
    # plt.savefig(f'../analysis/plots/deam_comparison/{songurl}/{datatype[:-1]}/{idx}.png')
    # plt.close()

    plt.show()
    
    if idx > 30:
        break
#%%
'''
plot histogram of percentage_match_list
'''
fig = plt.figure(figsize=(9,6))
plt.hist(percentage_match_list)
plt.ylabel('number of trials')
plt.xlabel('percentage match (%)')
plt.title(f'Histogram of how much COLLECTED labels fall within 1 std of DEAM labels for {datatype[:-1]}')
plt.savefig(f'../analysis/plots/deam_comparison/percentage_match_hist_{datatype[:-1]}.png')
plt.close()

# %%
'''
overall
'''
deamtrials = exps3[exps3['songurl'].str.contains('deam')]

for songurl, trials in deamtrials.groupby('songurl'):
    print(songurl)
    print(len(trials))
    ourlabels = []
    percentage_match_list = []
    for idx in trials.index:
        trial = trials.loc[idx]
        songurl = trial['songurl']
        # rescale to 0.5 and cut off 15 seconds in the front.
        ourlabel = average_1D(trial[datatype],5)[30::]
        ourlabels.append(ourlabel)
        temp = check_if_within_std(deamstats[songurl], ourlabel)
        percentage_match = round(sum(temp)/len(temp)*100, 2)
        percentage_match_list.append(percentage_match)
    
    ave_percentage_match = round(sum(percentage_match_list)/len(percentage_match_list), 2)
    plt = plot_collected_against_deam_all_participants(songurl, ourlabels, ave_percentage_match)
    # plt.show()
    plt.savefig(f'../analysis/plots/deam_comparison/all_together/{datatype[:-1]}_{songurl}.png')
    plt.close()

# %%
