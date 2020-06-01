import os
import sys
# sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath(''))
print(sys.path)
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import util
### to edit accordingly.
from dataloader import dataset_non_ave_with_profile as dataset_class
### to edit accordingly.
from network import Two_FC_layer as archi


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
print('cuda: ', use_cuda)
print('device: ', device)

# Parameters
affect_type = 'arousals'
params = {'batch_size': 128,
          'shuffle': True,c
          'num_workers': 6}
max_epochs = 10
# possible_conditions = np.array(['age', 'country_enculturation', 'country_live', 'fav_music_lang', 'gender', 'fav_genre', 'play_instrument', 'training', 'training_duration'])
conditions = ['age']


'''
load data
'''
feat_dict = util.load_pickle('data/feat_dict_ready.pkl')
exps = pd.read_pickle(os.path.join('data', 'exps_ready.pkl'))
pinfo = pd.read_pickle(os.path.join('data', 'pinfo_numero.pkl'))


dataset_obj = dataset_class(affect_type, feat_dict, exps, pinfo, conditions)
train_dataset = dataset_obj.gen_dataset(train=False)
train_loader = DataLoader(train_dataset, **params)
'''
testing dataloader
'''
# for data in train_loader:
#     print(data)
#     break

## MODEL
input_dim = 1582 + len(conditions)
model = archi(input_dim=input_dim).to(device)
model.float()
print(model)
model.train()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_epoch_log = []

for epoch in np.arange(max_epochs):
    start_time = time.time()
    loss_log = []

    # Training
    for batchidx, (feature, label) in enumerate(train_loader):
        numbatches = len(train_loader)
        # Transfer to GPU
        feature, label = feature.to(device).float(), label.to(device).float()
        # clear gradients 
        optimizer.zero_grad()
        # forward pass
        output = model.forward(feature)
        # MSE Loss calculation
        loss = criterion(output.squeeze(), label.squeeze())
        # backward pass
        loss.backward(retain_graph=True)
        # update parameters
        optimizer.step()
        # record training loss
        loss_log.append(loss.item())
        print(f'Epoch: {epoch} || Batch: {batchidx}/{numbatches} || MSELoss = {loss.item()}', end = '\r')
        
    aveloss = np.average(loss_log)
    print(' '*200)
    loss_epoch_log.append(aveloss)

    epoch_duration = time.time() - start_time
    print(f'Epoch: {epoch:3} || MSELoss: {aveloss:10.6f} || time taken (s): {epoch_duration}')


# plot loss against epochs
plt.plot(loss_epoch_log)
plt.xlabel('epoch')
plt.ylabel('mseloss')
plt.show()

# plot prediction of one song.
# feat_dict['']