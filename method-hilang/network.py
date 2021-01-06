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