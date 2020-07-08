import os
import sys
# sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath(''))
# print(sys.path)
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
from dataloader import dataset_non_ave_no_profile as dataset_class
### to edit accordingly.
from network import LSTM_single as archi


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
print('cuda: ', use_cuda)
print('device: ', device)

# Parameters
affect_type = 'arousals'
params = {'batch_size': 16,
          'shuffle': True,
          'num_workers': 6}
lstm_size = 10
step = 10
max_epochs = 100


'''
load data
'''
feat_dict = util.load_pickle('data/feat_dict_ready.pkl')
exps = pd.read_pickle(os.path.join('data', 'exps_ready.pkl'))
pinfo = pd.read_pickle(os.path.join('data', 'pinfo_numero.pkl'))

# standardize the features
def standardize(feat_dict): # all together
    mean_sum = np.zeros(1582)
    std_sum = np.zeros(1582)
    for songurl, audio_feat in feat_dict.items():
        # print(songurl)
        # print(np.shape(audio_feat))
        mean_sum += np.mean(audio_feat, axis=0)
        # print(np.shape(mean))
        std_sum += np.std(audio_feat,axis=0)
    
    mean = mean_sum/len(feat_dict)
    std = std_sum/len(feat_dict)

    for songurl, audio_feat in feat_dict.items():
        standard_feat = (audio_feat - mean)/std
        # print(standard_feat)
        feat_dict[songurl] = standard_feat

    return feat_dict


# print(feat_dict['deam_115'][0])
feat_dict = standardize(feat_dict)

def dataloader_prep(train=True):
    # prepare data for testing
    dataset_obj = dataset_class(affect_type, feat_dict, exps, lstm_size, step)
    dataset = dataset_obj.gen_dataset(train=train)
    loader = DataLoader(dataset, **params)

    return loader

## MODEL
input_dim = 1582 #+ len(conditions)
hidden_dim = 512
model = archi(input_dim, hidden_dim).to(device)
model.float()
print(model)
model.train()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

'''
testing model params.
'''




# for feature, label in test_loader:
#     feature, label = feature.to(device).float(), label.to(device).float()
    
#     feature = feature.permute(2,0,1)
#     print(feature.shape)
#     print(label.shape)
#     output = model(feature)
#     print(output.shape)
#     break


def train(train_loader):
    loss_epoch_log = []

    # one round without training.
    loss_log = []
    for batchidx, (feature, label) in enumerate(train_loader):
        numbatches = len(train_loader)
        # Transfer to GPU
        feature, label = feature.to(device).float(), label.to(device).float()
        feature = feature.permute(2,0,1)
        print('label shape:  ', label.shape)
        label = label.transpose(1,0)
        # clear gradients 
        optimizer.zero_grad()
        # forward pass
        output = model.forward(feature)
        # MSE Loss calculation
        loss = criterion(output.squeeze(), label.squeeze())
        loss_log.append(loss.item())
    aveloss = np.average(loss_log)
    loss_epoch_log.append(aveloss)
    print(f'Initial round without training || MSELoss = {aveloss:.6f}')

    for epoch in np.arange(1, max_epochs+1):
        start_time = time.time()
        loss_log = []

        # Training
        for batchidx, (feature, label) in enumerate(train_loader):
            numbatches = len(train_loader)
            # Transfer to GPU
            feature, label = feature.to(device).float(), label.to(device).float()
            feature = feature.permute(2,0,1)
            label = label.transpose(1,0)
            # clear gradients 
            model.zero_grad()
            # optimizer.zero_grad()
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
    plt.plot(loss_epoch_log[1::])
    plt.xlabel('epoch')
    plt.ylabel('mseloss')
    plt.title(f'Training loss (before training: {loss_epoch_log[0]:.6f})')
    dir_path = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(os.path.join(dir_path, 'saved_models', f'{model_name}_loss_plot.png'))
    
    plt.close()

    return model


def save_model(model, model_name):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_save = os.path.join(dir_path, 'saved_models', f"{model_name}.pth")
    torch.save(model.state_dict(), path_to_save)
    # loss_fig.savefig(os.path.join(args.model_path, f"{model_name}_loss_plot.png"))

def load_model(model_name):

    dir_path = os.path.dirname(os.path.realpath(__file__))
    path_to_load = os.path.join(dir_path, 'saved_models', f"{model_name}.pth")
    model.load_state_dict(torch.load(path_to_load))
    model.eval() # assuming loading for eval and not further training. (does not save optimizer so shouldn't continue training.)
    return model


def plot_pred_comparison(output, label, mseloss):
    plt.plot(output.cpu().numpy(), label='pred')
    plt.plot(label.cpu().numpy(), label='ori')
    plt.legend()
    plt.title(f'Prediction vs Ground Truth || mse: {mseloss}')
    return plt

def plot_pred_against(output, label, mseloss):
    actual = label.cpu().numpy()
    predicted = output.squeeze().cpu().numpy()
    print(np.shape(actual))
    print(np.shape(predicted))
    plt.scatter(actual, predicted)
    return plt


def single_test(model_name, index):
    # features - audio
    testfeat = feat_dict['00_145']
    # features - pinfo
    testtrial = exps[exps['songurl']=='00_145'].reset_index().loc[index]
    # labels
    testlabel = testtrial[affect_type]

    testinput = testfeat

    with torch.no_grad():
        testinput = torch.from_numpy(testinput)
        testlabel = torch.from_numpy(testlabel)

        feature, label = testinput.to(device).float(), testlabel.to(device).float()
        model = load_model(model_name)

        # forward pass
        output = model(feature)
        # MSE Loss calculation
        loss = criterion(output.squeeze(), label.squeeze())
    # print(loss.item())

    dir_path = os.path.dirname(os.path.realpath(__file__))
    
    plt = plot_pred_comparison(output.squeeze(), label, loss.item())
    plt.savefig(os.path.join(dir_path, 'saved_models', f'{model_name}_prediction_{index}.png'))
    plt.close()

    plt = plot_pred_against(output.squeeze(), label, loss.item())
    plt.savefig(os.path.join(dir_path, 'saved_models', f'{model_name}_y_vs_yhat_{index}.png'))
    plt.close()

def test(model):
    # prepare data for testing
    test_loader = dataloader_prep(train=False)

    loss_log = []
    with torch.no_grad():
        # model = load_model(model_name)
        for batchidx, (feature, label) in enumerate(test_loader):
            
            feature, label = feature.to(device).float(), label.to(device).float()
            # forward pass
            output = model(feature)
            # MSE Loss calculation
            loss = criterion(output.squeeze(), label.squeeze())
            # print(loss)
            loss_log.append(loss.item())
    aveloss = np.average(loss_log)
    print(f'average test lost (per batch): {aveloss}')

model_name = 'test'
loader = dataloader_prep(train=True)
model = train(loader)
# save_model(model, model_name)
# single_test(model_name,1)
# loader = dataloader_prep(train=False)
test(model)