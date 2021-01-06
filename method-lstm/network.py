'''
for lstm, averaged labels no profile...
'''

import torch
from torch import nn
from torch.nn import functional as F


class LSTM_single(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, drop_prob):
        super(LSTM_single, self).__init__()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        self.act1 = nn.LeakyReLU(0.1)

        self.fc = nn.Linear(hidden_dim, hidden_dim//2)
        self.dropout2 = nn.Dropout(drop_prob)
        self.act2 = nn.LeakyReLU(0.1)
        
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.dropout3 = nn.Dropout(drop_prob)
        self.act3 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(hidden_dim//4, 1)
        

    def forward(self, x):
        print(x.shape)
        lstm_out, lstm_h = self.lstm(x.view(len(x), -1, self.input_dim))
        lstm_out = self.dropout1(lstm_out)
        lstm_out = self.act1(lstm_out)
        
        out = self.fc(lstm_out.view(len(x),-1, self.hidden_dim))
        out = self.dropout2(out)
        out = self.act2(out)

        out = self.fc2(out)
        out = self.dropout3(out)
        out = self.act3(out)
        
        out = self.fc3(out)

        return out