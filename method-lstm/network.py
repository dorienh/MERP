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

        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
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
        # print('x shape: ', x.shape)
        lstm_out, lstm_h = self.lstm(x)
        # print('lstm_out shape: ', lstm_out.shape)
        lstm_out = self.dropout1(lstm_out)
        lstm_out = self.act1(lstm_out)
        
        # print(lstm_out.shape)
        out = self.fc(lstm_out)
        out = self.dropout2(out)
        out = self.act2(out)
        # print('fc1 shape: ', out.shape)

        out = self.fc2(out)
        out = self.dropout3(out)
        out = self.act3(out)
        # print('fc2 shape: ', out.shape)
        
        out = self.fc3(out)
        # print('out shape: ', out.shape)
        return out


if __name__ == "__main__":

    train_feat_dict = util.load_pickle('data/train_feats_pca_windowed.pkl')
    test_feat_dict = util.load_pickle('data/test_feats_pca_windowed.pkl')
    exps = pd.read_pickle('data/exps_std_a_ave_windowed.pkl')

    ## MODEL
    input_dim = 724 # 1582 
    model = archi(input_dim=input_dim, hidden_dim=args.hidden_dim, drop_prob=args.drop_prob).to(device)
    model.float()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ########################
    ####    Training    ####
    ########################
     
    train_loader = dataloader_prep(train_feat_dict, exps, args, train=True)
    test_loader = dataloader_prep(test_feat_dict, exps, args, train=False)
    
    model, test_ave_mse, test_ave_r = train(train_loader, model, test_loader, args)