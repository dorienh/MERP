# %%
# heat map??
import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))
print(sys.path)
import util
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import torch

torch.cuda.is_available()

#%%
temp = torch.load('../method-hilang/saved_models/a_np_5fold/a_np_5fold_0.pth')
#%%

alphas_a = util.load_pickle('../plots/krippendorff/alphas_a.pkl')
alphas_v = util.load_pickle('../plots/krippendorff/alphas_v.pkl')

print(alphas_a)
print(alphas_v)

# %%
affect_type = 'valences'
exps = pd.read_pickle('../../data/exps_ready3.pkl')
ave_exps = pd.read_pickle('../../data/exps_std_a_ave3.pkl')
ave_exps_age = pd.read_pickle(f'../../data/exps_std_a_profile_ave_age.pkl')

#%%
for idx, row in ave_exps_age.iterrows():
    plt.plot(row['labels'])
    plt.show()
    

#%%
prof_exp_log_profile = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg-prof/saved_models/experiment_log2.pkl')
prof_exp_log_lstm = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg/saved_models/experiment_log3.pkl')
prof_exp_log_linear = pd.read_pickle('/home/meowyan/Documents/emotion/method-hilang/saved_models/experiment_log3.pkl')

 # %%
for songurl, group in exps.groupby('songurl'):
    arousal_series = group[affect_type].to_numpy()
    # valence_series = group['valences'].to_numpy()
    worker_series = group['workerid'].to_numpy()
    # temp = [print(worker) for label,worker in zip(label_series, worker_series) ]
    song_df = pd.DataFrame.from_records(arousal_series)

    fig = plt.figure(figsize=(20,10))

    sns.heatmap(song_df, vmin=-1, vmax=1)
    plt.savefig(f'../plots/heatmaps/{affect_type}/{songurl}_partcipant_time.png')
    
    

# %%
exp_log_filepath = os.path.join(dir_path,'saved_models','experiment_log2.pkl')
pd.read_pickle(exp_log_filepath)


#%%
# EXTRA - find the total length of the 54 songs put together. min, max and mean lengths too.
import os
import sys
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('..'))
import util

import librosa

songfilenamedict = {}

filepath = '/home/meowyan/Documents/emotion/data/50songs'
for e in util.songlist:
    one, two = e.split('_')
    song_filepath = os.path.join(filepath, one, f'{two}.wav')
    songfilenamedict[e] = song_filepath
    # this is inefficient but... i get to see the time of each song so maybe it's worth it lol
songlengthdict = {}
for key, path in songfilenamedict.items():
    y, sr = librosa.load(path)
    length = round(len(y)/ sr *10, 2)
    print(length)
    songlengthdict[key] = length
    



# %%
import numpy as np

lengths = np.array(list(songlengthdict.values()))
print('min length: ', min(lengths))
print('max length: ', max(lengths))
print('mean length: ', np.mean(lengths))
print('sum length: ', sum(lengths))
'''
00 sum: 20462.82 (34.1 mins)
01 sum: 13775.09 (22.96 mins)
0505 sum: 18870.58 (31.45 mins)
10 sum: 9622.18 (16.04 mins)
11 sum: 23245.159 (38.74 mins)
'''
# %%

# total number of datapoints after pruning
arousals = exps.arousals
count = 0
for idx, arousal_list in arousals.iteritems():
    count += len(arousal_list)
print(count)

# total number of datapoints after averaging
count = 0
for idx, row in ave_exps.iterrows():
    count += len(row['labels'])
print(count)

# %%
import librosa


# y, sr = librosa.load('/Users/koen/Desktop/Amazon/50songs/11/748.wav',duration=30)
y, sr = librosa.load('../../data/50songs/00/366.wav',duration=30)

import librosa.display

fig = plt.figure(figsize=(20,4))
librosa.display.waveplot(y, sr=sr)

plt.savefig('../plots/example_waveform.png')

temp = exps[exps.songurl.str.contains('00_366')]
valences = temp['valences'].mean()
arousals = temp['arousals'].mean()

fig = plt.figure(figsize=(20,4))
plt.plot(arousals[0:60])
plt.xlabel('time')
plt.ylabel('arousal')
plt.xticks(np.arange(0, 60, 10), np.arange(0,30,5))
plt.savefig('../plots/example_waveform_a.png')
# plt.ylim(-1,1)

fig = plt.figure(figsize=(20,4))
plt.plot(valences[0:60])
plt.xlabel('time')
plt.ylabel('valence')
plt.xticks(np.arange(0, 60, 10), np.arange(0,30,5))
plt.savefig('../plots/example_waveform_v.png')
# %%

'''
all the profile distributions lets go
'''
plt.rcParams.update({'font.size': 8})
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"

pinfo = pd.read_pickle('../../data/mediumrare/semipruned_pinfo.pkl')
pinfo_numero = pd.read_pickle('../../data/pinfo_numero.pkl')
print(len(pd.unique(pinfo.workerid)))
print(len(pd.unique(exps.workerid)))

pinfo = pinfo.loc[pinfo.workerid.str.contains('|'.join(exps.workerid))].reset_index(drop=True)
pinfo = pinfo.drop(columns='batch')
pinfo_numero = pinfo_numero.loc[pinfo_numero.workerid.str.contains('|'.join(exps.workerid))].reset_index(drop=True)

#%%
# age, gender, master, country_enculturation, country_live, fav_music_lang, fav_genre, play_instrument, training, training_duration

age = pinfo['age']
numages = age.nunique()
fig = plt.figure(figsize=(10,5))
plt.hist(age,bins=numages)
plt.xlabel('Age')
plt.ylabel('Number of Participants')
plt.tight_layout()
plt.savefig('../plots/pinfo/age.png')

#%%
age_cat = pinfo_numero['age']
numages = age_cat.nunique()
fig = plt.figure(figsize=(10,5))
plt.hist(age_cat,bins=numages)
# range_map = {(0,25):-1.0, (26,35):-0.33, (36,50):0.33, (51,80):1.0}
# range_map = {(0,25):0.0, (26,35):0.33, (36,50):0.66, (51,80):1.0}
plt.xticks([0.0,0.25,0.5,0.75,1.0], [0,26,36,51,80])
plt.xlabel('Age')
plt.ylabel('Number of Participants')
plt.tight_layout()
plt.savefig('../plots/pinfo/age_binned.png')

#%%
# combine age plots, normal and binned 
age = pinfo['age']
numages = age.nunique()
age_cat = pinfo_numero['age']
numages_cat = age_cat.nunique()
gender = pinfo['gender']
sizes = gender.value_counts()

fig = plt.figure(figsize=(5.2,3), dpi=300)
grid = plt.GridSpec(2,2)

# plt.ylabel('Number of Participants')
plt.subplot(grid[0,0])
plt.hist(age,bins=numages)
plt.title('(A) Age')
plt.subplot(grid[1,0])
plt.hist(age_cat,bins=numages_cat)
plt.xticks([0.0,0.25,0.5,0.75,1.0], [0,26,36,51,80])
plt.title('(B) Binned Age')
# plt.ylabel('Number of Participants')
plt.tight_layout()
fig.text(0.005, 0.5, 'Number of Participants', va='center', rotation='vertical')



plt.subplot(grid[0:,1])
# print(sizes)
# fig = plt.figure(figsize=(8,8))
sizes.plot.pie(autopct='%1.1f%%', colors=['tab:blue', 'tab:purple', 'tab:olive'], title="(C) Gender")
plt.ylabel('')
plt.tight_layout()
# plt.savefig('../plots/pinfo/age_gender_combined.png')

# %%

# gender_cat = pinfo_numero['gender']
gender = pinfo['gender']
# labels = gender.unique()
sizes = gender.value_counts()
print(sizes)
fig = plt.figure(figsize=(8,8))
sizes.plot.pie(autopct='%1.1f%%', colors=['tab:blue', 'tab:purple', 'tab:olive'])
plt.savefig('../plots/pinfo/gender.png')



# %%
master_cat = pinfo_numero['master']
master = pinfo['master']
# labels = gender.unique()
sizes = master.value_counts()
print(sizes)
fig = plt.figure(figsize=(8,8))
sizes.plot.pie(autopct='%1.1f%%',labels=['No', 'Yes'],colors=['tab:blue', 'tab:purple', 'tab:olive'])
plt.savefig('../plots/pinfo/master.png')

#%%
country_r = pinfo_numero['country_live']
sizes_r = country_r.value_counts()
country_e = pinfo_numero['country_enculturation']
sizes_e = country_e.value_counts()

fig = plt.figure(figsize=(5,3), dpi=300)
plt.subplot(1,2,1)
sizes_r.plot.pie(autopct='%1.1f%%',labels=['USA', 'India', 'Others'],colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])

plt.subplot(1,2,2)
sizes_e.plot.pie(autopct='%1.1f%%',labels=['USA', 'India', 'Others'],colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])

plt.tight_layout()
plt.savefig('../plots/pinfo/country_r_e.png')

# %%
country_e = pinfo['country_enculturation']
sizes = country_e.value_counts()
print(sizes)
fig = plt.figure(figsize=(3,3), dpi=300)
sizes.plot.pie(autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.25)
# sizes.plot.pie(autopct='%1.1f%%',labels=['USA', 'India', 'Others'],colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])
# plt.savefig('../plots/pinfo/country_e.png')
plt.ylabel('')
plt.title('Country of music enculturation')
# plt.tight_layout()
print(pinfo['country_enculturation'].nunique()) #15
#['US' 'JP' 'IN' 'EC' 'MX' 'IT' 'ZA' 'RU' 'GB' 'AM' 'CO' 'AS' 'NZ' 'AE' 'BR']
# US
# Japan
# India
# Ecuador
# Mexico
# Italy
# South Africa
# Russia
# Great Britain
# Armenia
# Colombia
# American Samoa
# New Zealand
# United Arab Emirates
# Brazil
# %%
country_r = pinfo_numero['country_live']
sizes = country_r.value_counts()
print(sizes)
fig = plt.figure(figsize=(8,8))
sizes.plot.pie(autopct='%1.1f%%',labels=['USA', 'India', 'Others'],colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'], pctdistance=1.1)
plt.savefig('../plots/pinfo/country_r.png')
print(pinfo['country_live'].nunique()) #11
# ['US' 'IN' 'IT' 'ZA' 'RU' 'ID' 'GB' 'AM' 'AS' 'RO' 'BR']
# US
# India
# Italy
# South Africa
# Russia
# Indonesia
# Great Britain
# Armenia
# American Samoa
# Romania
# Brazil
#%%
fav_music_lang = pinfo_numero['fav_music_lang']
sizes_l = fav_music_lang.value_counts()

sizes_g = pinfo_numero['fav_genre'].value_counts()

fig = plt.figure(figsize=(5,3), dpi=300)
plt.subplot(1,2,1)
sizes_l.plot.pie(autopct='%1.1f%%',labels=['English', 'Tamil', 'Others'], colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])

plt.subplot(1,2,2)
sizes_g.plot.pie(autopct='%1.1f%%',labels=['Others', 'Rock', 'Classical', 'Pop'], colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])

plt.tight_layout()
plt.savefig('../plots/pinfo/music_listening.png')

# %%

fav_music_lang = pinfo['fav_music_lang']
print(pinfo['fav_music_lang'].nunique())  #11
# ['EN', 'KO', 'JA', 'TE', 'TA', 'HI', 'ML', 'IT', 'HY', 'DE', 'BN']
# English
# Korean
# Japanese
# Telugu
# Tamil
# Hindi
# Malayalam
# Italian
# Armenian
# German
# Bengali

# sizes = pinfo['fav_music_lang'].value_counts()
# sizes.plot.pie(colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:brown'])
sizes = fav_music_lang.value_counts()
print(sizes)
fig = plt.figure(figsize=(3,3),dpi=300)

sizes.plot.pie(autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.25)
plt.title('Preferred language of music lyrics')
plt.ylabel('')
# sizes.plot.pie(autopct='%1.1f%%',labels=['English'/, 'Tamil', 'Others'], colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])
# plt.savefig('../plots/pinfo/fav_music_lang.png')
# %%
# fav_genre
sizes1 = pinfo_numero['fav_genre'].value_counts()
sizes2 = pinfo['fav_genre'].value_counts()
print(sizes1)
fig = plt.figure(figsize=(3,3), dpi=300)
sizes2.plot.pie(autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.25)
plt.title('Preferred genre')
plt.ylabel('')
# sizes1.plot.pie(autopct='%1.1f%%',labels=['Others', 'Rock', 'Classical', 'Pop'], colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])
pinfo['fav_genre'].nunique() #13
#['Pop', 'Country', 'Metal', 'Rhythm and Blues', 'Rock', 'Jazz', 'Other', 'Indie Rock', 'Electronic dance music', 'Electro', 'Dubstep', 'Classical music', 'Techno']
# plt.savefig('../plots/pinfo/fav_genre.png')
# %%
# play_instrument
sizes = pinfo['play_instrument'].value_counts()
print(sizes)
fig = plt.figure(figsize=(8,8))
sizes.plot.pie(autopct='%1.1f%%',colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])
plt.savefig('../plots/pinfo/play_instrument.png')
#%%
# training
sizes = pinfo['training'].value_counts()
print(sizes)
fig = plt.figure(figsize=(8,8))
sizes.plot.pie(autopct='%1.1f%%',colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])
plt.savefig('../plots/pinfo/training.png')
#%%
# training_duration

sizes = pinfo['training_duration'].value_counts()
sizes1 = pinfo_numero['training_duration'].value_counts()
# print(sizes)
print(sizes1)
fig = plt.figure(figsize=(8,8))
sizes1.plot.pie(autopct='%1.1f%%',labels=['[1-5]','0' , '>6'],colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])
plt.savefig('../plots/pinfo/training_duration.png')
# %%
#%%
sizes_pi = pinfo['play_instrument'].value_counts()
sizes_tr = pinfo['training'].value_counts()
sizes_td = pinfo_numero['training_duration'].value_counts()
sizes_m = pinfo['master'].value_counts()

fig = plt.figure(figsize=(5,6),dpi=300)
plt.subplot(221)
sizes_pi.plot.pie(autopct='%1.1f%%',colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])
plt.subplot(222)
sizes_tr.plot.pie(autopct='%1.1f%%',colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])
plt.subplot(223)
sizes_td.plot.pie(autopct='%1.1f%%',labels=['[1-5]','0' , '>6'],colors=['tab:blue', 'tab:purple', 'tab:olive', 'tab:cyan'])
plt.subplot(224)
sizes_m.plot.pie(autopct='%1.1f%%',labels=['No', 'Yes'],colors=['tab:blue', 'tab:purple', 'tab:olive'])
plt.subplots_adjust(left=0.6,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.001, 
                    hspace=0.001)
plt.tight_layout()
plt.savefig('../plots/pinfo/music_exp_master.png')
#%%
## num participants label each song

temp = exps.groupby('songurl')['workerid'].nunique()


# %%

import torch
a = torch.tensor([0.0434, 0.0771, 0.1024, 0.1464, 0.1504, 0.1788, 0.1907, 0.1423, 0.1055, 0.1002])

b = torch.tensor([0.0877, 0.0877, 0.0877, 0.0877, 0.0877, 0.0877, 0.0877, 0.0877, 0.0877, 0.0877])



def pearson_corr_loss(output, target):
    x = output
    y = target

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    print(torch.sum(vx ** 2))
    print(torch.sum(vy ** 2))

    cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return cost*-1

temp = pearson_corr_loss(a, b)
print(temp)
# %%
