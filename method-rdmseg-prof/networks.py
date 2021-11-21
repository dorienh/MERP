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

class lstm_single(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_single, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # self.attn = MutliHeadAttention1D(in_features, out_features, kernel_size)

        self.dropout1 = nn.Dropout(0.5)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(hidden_dim, 1)
        

    def forward(self, x):
        # print('forward')

        lstm_out, _ = self.lstm(x)
        # print(lstm_out.narrow(0,1,0))
        # print(f'lstm weight max = {self.lstm.weight_ih_l0.max()}\tweight min = {self.lstm.weight_ih_l0.min()}\tmean = {self.lstm.weight_ih_l0.mean()}')
        
        lstm_out = self.dropout1(lstm_out)
        lstm_out = self.act1(lstm_out)
        # print(lstm_out)

        out = self.fc3(lstm_out)
        # print(out)
        # print('out shape: ', out.shape)
        return out

class lstm_single_2fc(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(lstm_single_2fc, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        # self.attn = MutliHeadAttention1D(in_features, out_features, kernel_size)

        self.dropout1 = nn.Dropout(0.5)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)

        self.dropout2 = nn.Dropout(0.5)
        self.act2 = nn.LeakyReLU(0.1)

        self.fc2 = nn.Linear(hidden_dim//2, 1)
        

    def forward(self, x):
        # print('forward')

        lstm_out, _ = self.lstm(x)
        # print(lstm_out.narrow(0,1,0))
        # print(f'lstm weight max = {self.lstm.weight_ih_l0.max()}\tweight min = {self.lstm.weight_ih_l0.min()}\tmean = {self.lstm.weight_ih_l0.mean()}')
        
        lstm_out = self.dropout1(lstm_out)
        lstm_out = self.act1(lstm_out)
        # print(lstm_out)

        out = self.fc1(lstm_out)

        out = self.dropout2(out)
        out = self.act2(out)

        out = self.fc2(out)

        # print(out)
        # print('out shape: ', out.shape)
        return out

class Three_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 724, hidden_dim=128):
    # def __init__(self, input_dim = 1582, reduced_dim=128, fc_dim = 64):
        super(Three_FC_layer, self).__init__()
        # self.reduce_dim = nn.Linear(input_dim, reduced_dim, bias=False)

        self.fc1 = nn.Linear(input_dim, hidden_dim)#, bias=False)
        self.dropout1 = nn.Dropout(0.5)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)#, bias=False)
        self.dropout2 = nn.Dropout(0.5)
        self.lr2 = nn.LeakyReLU(0.1)
        self.fc_out = nn.Linear(hidden_dim//2, out_features=1)#, bias=False)  # output
        self.actout = nn.Tanh()

        # kernel = torch.FloatTensor([[[0.006, 0.061, 0.242, 0.383, 0.242, 0.061, 0.006]]]) # sigma = 1
        kernel = torch.FloatTensor([[[0.0099, 0.0301, 0.0587, 0.0733, 0.0587, 0.0301, 0.0099]]]) # sigma = 1.5, kernel size = 7

        self.register_buffer('kernel', kernel)

    def forward(self, x):
        # out = self.class_dim(self.lr2(self.dropout2(self.fc2(self.lr1(self.dropout1(self.fc1(self.reduced_rgb(x))))))))
        # out = self.reduce_dim(x)
        out = self.fc1(x)
        # print(out)
        # print(self.fc1.weight.grad)
        # print(self.fc1.weight)
        # print(f'weight max = {self.fc1.weight.max()}\tweight min = {self.fc1.weight.min()}\tmean = {self.fc1.weight.mean()}')
        
        # print('1 ', x.shape)
        out = self.dropout1(out)
        out = self.lr1(out)
        out = self.fc2(out)
        # print('2 ', out.shape)
        out = self.dropout2(out)
        out = self.lr2(out)
        out = self.fc_out(out)
        out = self.actout(out)
        # print('3 ', out.shape)
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

# class Three_FC_layer(torch.nn.Module):
#     def __init__(self, input_dim = 261, reduced_dim=512, fc_dim = 64):
#     # def __init__(self, input_dim = 1582, reduced_dim=128, fc_dim = 64):
#         super(Three_FC_layer, self).__init__()
#         # self.reduce_dim = nn.Linear(input_dim, reduced_dim, bias=False)

#         self.fc1 = nn.Linear(reduced_dim, fc_dim, bias=False)
#         self.dropout1 = nn.Dropout(0.5)
#         self.lr1 = nn.LeakyReLU(0.1)
#         self.fc2 = nn.Linear(fc_dim, fc_dim, bias=False)
#         self.dropout2 = nn.Dropout(0.5)
#         self.lr2 = nn.LeakyReLU(0.1)
#         self.fc_out = nn.Linear(fc_dim, out_features=1, bias=False)  # output

#     def forward(self, x):
#         # out = self.class_dim(self.lr2(self.dropout2(self.fc2(self.lr1(self.dropout1(self.fc1(self.reduced_rgb(x))))))))
#         # out = self.reduce_dim(x)
#         out = self.fc1(x)
#         out = self.dropout1(out)
#         out = self.lr1(out)
#         out = self.fc2(out)
#         out = self.dropout2(out)
#         out = self.lr2(out)
#         out = self.fc_out(out)
        
#         return out
