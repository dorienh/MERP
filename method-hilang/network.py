'''
I don't know what I'm doing lol.
'''

import torch
from torch import nn
from torch.nn import functional as F



##############################################################
####                  1) Fully Connected                  ####
##############################################################

class Two_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 724, reduced_dim=128, fc_dim = 64):
    # def __init__(self, input_dim = 1582, reduced_dim=128, fc_dim = 64):
        super(Two_FC_layer, self).__init__()
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

##############################################################
####                  2) Fully Connected                  ####
##############################################################

class Three_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 724, reduced_dim=128):
    # def __init__(self, input_dim = 1582, reduced_dim=128, fc_dim = 64):
        super(Three_FC_layer, self).__init__()
        # self.reduce_dim = nn.Linear(input_dim, reduced_dim, bias=False)

        self.fc1 = nn.Linear(input_dim, reduced_dim, bias=True)
        self.dropout1 = nn.Dropout(0.5)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(reduced_dim, reduced_dim//2, bias=True)
        self.dropout2 = nn.Dropout(0.5)
        self.lr2 = nn.LeakyReLU(0.1)
        self.fc_out = nn.Linear(reduced_dim//2, out_features=1, bias=True)  # output
        
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