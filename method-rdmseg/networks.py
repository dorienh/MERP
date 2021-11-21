import os
import sys
# sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath(''))
print(sys.path)
import util

import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F

from rdm_dataset import rdm_dataset 
from torch.utils.data import DataLoader


class lstm_single(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_single, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # self.attn = MutliHeadAttention1D(in_features, out_features, kernel_size)

        self.dropout1 = nn.Dropout(0.5)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(hidden_dim, 1)
        

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout1(lstm_out)
        lstm_out = self.act1(lstm_out)

        out = self.fc3(lstm_out)
        # print('out shape: ', out.shape)
        return out


class lstm_double(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_double, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # self.attn = MutliHeadAttention1D(in_features, out_features, kernel_size)

        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, batch_first=True, bidirectional=True)

        self.fc2 = nn.Linear(hidden_dim*2, 1)

        # self.dropout2 = nn.Dropout(0.2)
        self.actout = nn.Tanh()
        
        # smoothing gaussian kernel
        # kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]) # sigma = 1
        # kernel = torch.FloatTensor([[[0.0099, 0.0301, 0.0587, 0.0733, 0.0587, 0.0301, 0.0099]]]) # sigma = 1.5, kernel size = 7

        # self.register_buffer('kernel', kernel) # to device workaround
        

    def forward(self, x):
        # print('forward')

        lstm_out, lstm_c = self.lstm(x)
        # print(lstm_out.narrow(0,1,0))
        # print(f'lstm weight max = {self.lstm.weight_ih_l0.max()}\tweight min = {self.lstm.weight_ih_l0.min()}\tmean = {self.lstm.weight_ih_l0.mean()}')

        lstm_out, _ = self.lstm2(lstm_out, lstm_c)
        
        # out = self.dropout2(lstm_out)

        out = self.fc2(lstm_out)
        
        out = self.actout(out)
        # print('3 ', out.shape)
        out = out.flatten(1) # remove last dimension [8,10,1]
        out = out.unsqueeze(1) # create channel dim for conv [8,1,10]

        # Apply smoothing
        # out = F.conv1d(out, self.kernel, padding=3)

        # print(out)
        # print('out shape: ', out.shape)
        return out
    
    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

class Three_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 724, hidden_dim=128):
    # def __init__(self, input_dim = 1582, reduced_dim=128, fc_dim = 64):
        super(Three_FC_layer, self).__init__()
        # self.reduce_dim = nn.Linear(input_dim, reduced_dim, bias=False)

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.dropout1 = nn.Dropout(0.5)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2, bias=True)
        self.dropout2 = nn.Dropout(0.5)
        self.lr2 = nn.LeakyReLU(0.1)
        self.fc_out = nn.Linear(hidden_dim//2, out_features=1, bias=True)  # output
        
        self.actout = nn.Tanh()

        # Create gaussian kernels
        # kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]])
        # kernel = torch.FloatTensor([[[0.0012, 0.0025, 0.0045, 0.0074, 0.0110, 0.0145, 0.0171, 0.0181, 0.0171, 0.0145, 0.0110, 0.0074, 0.0045, 0.0025, 0.0012]]])
        kernel = torch.FloatTensor([[[0.0099, 0.0301, 0.0587, 0.0733, 0.0587, 0.0301, 0.0099]]]) # sigma = 1.5, kernel size = 7

        self.register_buffer('kernel', kernel)

    def forward(self, x):
        # out = self.class_dim(self.lr2(self.dropout2(self.fc2(self.lr1(self.dropout1(self.fc1(self.reduced_rgb(x))))))))
        # out = self.reduce_dim(x)
        out = self.fc1(x)
        # print('1 ', x.shape)
        out = self.dropout1(out)
        out = self.lr1(out)
        out = self.fc2(out)
        # print('2 ', out.shape)
        out = self.dropout2(out)
        out = self.lr2(out)
        out = self.fc_out(out)
        # print('3 ', out.shape)
        out = self.actout(out)
        
        out = out.flatten(1)
        out = out.unsqueeze(1)
        # Apply smoothing
        out = F.conv1d(out, self.kernel, padding=3)
        return out

    def load_my_state_dict(self, state_dict):
 
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

class conv_lstm(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super(conv_lstm, self).__init__()
        
        padding = kernel_size//2
        self.input_dim = input_dim


        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim//4, batch_first=True)
        # self.attn = MutliHeadAttention1D(in_features, out_features, kernel_size)

        # self.dropout1 = nn.Dropout(drop_prob)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(hidden_dim//4, 1)
        

    def forward(self, x):
        x = x.transpose(1,2)
        out = self.conv1d(x)
        # print(out.shape)

        out = out.transpose(1,2)
        lstm_out, _ = self.lstm(out)

        lstm_out = self.act1(lstm_out)

        out = self.fc3(lstm_out)
        # print('out shape: ', out.shape)
        return out


class Four_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 724, reduced_dim=128, fc_dim = 64):
    # def __init__(self, input_dim = 1582, reduced_dim=128, fc_dim = 64):
        super(Four_FC_layer, self).__init__()
        self.reduced_rgb = nn.Linear(input_dim, reduced_dim, bias=False)

        self.fc1 = nn.Linear(reduced_dim, fc_dim, bias=False)
        self.dropout1 = nn.Dropout(0.1)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(fc_dim, fc_dim, bias=False)
        self.dropout2 = nn.Dropout(0.5)
        self.lr2 = nn.LeakyReLU(0.1)
        self.class_dim = nn.Linear(fc_dim, out_features=1, bias=False)  # output

    def forward(self, x):
        out = self.class_dim(self.lr2(self.dropout2(self.fc2(self.lr1(self.dropout1(self.fc1(self.reduced_rgb(x))))))))
        return out




if __name__ == "__main__":

    train_feat_dict = util.load_pickle('data/train_feats.pkl')
    test_feat_dict = util.load_pickle('data/test_feats.pkl')
    exps = pd.read_pickle('data/exps_std_a_ave.pkl')

    # print(train_feat_dict['11_528'].shape[1])

    ## MODEL
    device = 'cuda'
    input_dim = list(train_feat_dict.values())[0].shape[1]# 1582 
    model = lstm_double(input_dim=input_dim, hidden_dim=64).to(device)
    print('num params: ', util.count_parameters(model))

    '''
    model.float()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    ########################
    ####    Training    ####
    ########################
     
    train_dataset = rdm_dataset(train_feat_dict, exps)
    test_dataset = rdm_dataset(test_feat_dict, exps)

    params = {'batch_size': 8, # must be smaller than number of songs in training set. (39)
            'shuffle': True,
            'num_workers': 10}
    # prepare data for testing
    train_loader = DataLoader(train_dataset, **params)
    
    for batchidx, (feature, label) in enumerate(train_loader):
        print(batchidx)
        numbatches = len(train_loader)
        # Transfer to GPU
        feature, label = feature.to(device).float(), label.to(device).float()
        print(feature.shape)
        print(label.shape)
        output = model.forward(feature)

        print(output.shape)

    # x = torch.tensor((((1,2,7),(3,4,8),(5,6,9)),((1,2,7),(3,4,8),(5,6,9))))
    # print(x.shape)
    # print(x.view(3,3,2))
    # print(x.transpose(1,2))
    # print(x.transpose(2,1))

    '''