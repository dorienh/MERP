#%%
import os
import sys
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('..'))

import util
from ave_exp_by_prof import ave_exps_by_profile

import pandas as pd
import numpy as np
import torch


feat_dict = util.load_pickle('../data/feat_dict_ready2.pkl')
exps = pd.read_pickle(os.path.join('..','data', 'exps_ready3.pkl'))
# %%
arousals = exps['arousals']
# %%
count = 0
for index, row in arousals.iteritems():
    count += len(row)
    
print(count) # 783498
# %%
# count should be the same as feat_dict count since we average all trials of the same song.
exps_ave = pd.read_pickle(os.path.join('..','data', 'exps_std_a_ave3.pkl'))
aved_count = 0
for index, row in exps_ave.iterrows():
    aved_count += len(row['labels'])
print(aved_count)

feat_count = 0

for k,v in feat_dict.items():
    feat_count += len(v)
print(feat_count)

# %%
affect_type = 'arousals'
profile = ['age', 'gender']
# profile = ['age']
pinfo = util.load_pickle('../data/pinfo_numero.pkl')

exps_prof_aved = ave_exps_by_profile(exps, pinfo, affect_type, profile)
prof_aved_count = 0
for index, row in exps_prof_aved.iterrows():
    prof_aved_count += len(row['labels'])
print(prof_aved_count)
# %%
import torch

def pearson_corr_loss(output, target):
        x = output
        y = target

        vx = x - torch.mean(x)
        print(torch.mean(x).shape)
        vy = y - torch.mean(y)
        print(torch.mean(y).shape)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        if torch.isnan(cost):
            return torch.tensor([0]).to(device)
        else:
            return cost*-1

output = torch.randn(8,10)
target = torch.randn(8,10)

pearson_corr_loss(output,target)
# %%
import torch
def pearson_corr_loss(output, target, reduction='mean'):
        x = output
        y = target

        vx = x - x.mean(1).unsqueeze(-1) # Keep batch, only calcuate mean per sample
        vy = y - y.mean(1).unsqueeze(-1)

        cost = (vx * vy).sum(1) / (torch.sqrt((vx ** 2).sum(1)) * torch.sqrt((vy ** 2).sum(1)))
        # cost = torch.nan_to_num(cost) # doesn't exist.... for some reason.
        cost[torch.isnan(cost)] = torch.tensor([0]).to(device)
        # reducing the batch of pearson to either mean or sum
        if reduction=='mean':
            return cost.mean()
        elif reduction=='sum':
            return cost.sum()
        elif reduction==None:
            return cost
# %%
output = torch.randn(8,10)
target = torch.randn(8,10)

pearson_corr_loss(output,target)
# %%

# %%
