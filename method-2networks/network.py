import torch
from torch import nn
from torch.nn import functional as F
import torch


##############################################################
####                  1) Fully Connected                  ####
##############################################################

class Two_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 1582, reduced_dim=128, fc_dim = 64):
        super(Two_FC_layer, self).__init__()
        self.reduced_rgb = nn.Linear(input_dim, reduced_dim, bias=False)

        self.fc1 = nn.Linear(reduced_dim, fc_dim, bias=False)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(fc_dim, fc_dim, bias=False)
        self.lr2 = nn.LeakyReLU(0.1)
        self.class_dim = nn.Linear(fc_dim, out_features=1, bias=False)  # output

    def forward(self, x):
        out = self.class_dim(self.lr2(self.fc2(self.lr1(self.fc1(self.reduced_rgb(x))))))
        return out

##############################################################
####            2) Combine 2 features network             ####
##############################################################


class Combination_model_1(torch.nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, fc_input_dim, fc_output_dim, lstm_size, drop_prob):
        super(Combination_model_1, self).__init__()
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim)
        self.dropout1 = nn.Dropout(drop_prob)

        self.fc_a1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2)
        self.sigmoid = nn.Sigmoid()
        # self.lr1 = nn.LeakyReLU(0.1)
        self.fc_a2 = nn.Linear(lstm_hidden_dim//2, lstm_hidden_dim//4)
        self.lr2 = nn.LeakyReLU(0.1)
        
        self.fc_a3 = nn.Linear(lstm_hidden_dim//4, 1)
        self.lr3 = nn.LeakyReLU(0.1)

        self.fc_p1 = nn.Linear(fc_input_dim, fc_output_dim)

        self.fc_c1 = nn.Linear(lstm_size + fc_output_dim, 1) #lstm_size)
        
        # input_dim = lstm_hidden_dim//4 + fc_output_dim
        # self.fc_c1 = nn.Linear(input_dim, input_dim//2)
        # self.lr3 = nn.LeakyReLU(0.1)
        # self.fc_c2 = nn.Linear(input_dim//2, input_dim//4)
        # self.lr4 = nn.LeakyReLU(0.1)
        # self.fc_c3 = nn.Linear(input_dim//4, 1)
    
    def forward(self, x_a, x_p):

        lstm_out, lstm_h = self.lstm(x_a.view(len(x_a), -1, self.lstm_input_dim))
        lstm_out = self.dropout1(lstm_out)

        hl_l = self.fc_a1(lstm_out.view(len(x_a), -1, self.lstm_hidden_dim))
        hl_l = self.sigmoid(hl_l)

        hl_l = self.fc_a2(hl_l)
        hl_l = self.lr2(hl_l)

        hl_l = self.fc_a3(hl_l)
        hl_l = self.lr3(hl_l)

        hl_p = self.fc_p1(x_p)

        # print('hidden layer 2: ', hl_3.squeeze().shape)
        # print('hidden_layer p: ', hl_p.shape)
        
        combined = torch.cat((hl_l.view(hl_l.size(0), -1), hl_p), dim=1)
        # fullyconnected([10 timesteps] + [1 profile information])???

        # print('combined.shape: ', combined.shape)

        c_1 = self.fc_c1(combined)
        # c_2 = self.fc_c2(c_1)
        # c_3 = self.fc_c3(c_2)
        

        return c_1



class Combination_model_2(torch.nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, fc_input_dim, fc_output_dim, lstm_size):
        super(Combination_model_2, self).__init__()
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim)

        self.fc_a1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc_a2 = nn.Linear(lstm_hidden_dim//2, lstm_hidden_dim//4)
        self.lr2 = nn.LeakyReLU(0.1)
        # self.fc_a3 = nn.Linear(lstm_hidden_dim//4, lstm_hidden_dim//8)
        # self.lr3 = nn.LeakyReLU(0.1)

        self.fc_p1 = nn.Linear(fc_input_dim, fc_output_dim)
        self.lr_p1 = nn.LeakyReLU(0.1)
        
        input_dim = lstm_size*lstm_hidden_dim//4 + fc_output_dim
        self.fc_c1 = nn.Linear(input_dim, input_dim//2)
        self.lr3 = nn.LeakyReLU(0.1)
        self.fc_c2 = nn.Linear(input_dim//2, input_dim//4)
        self.lr4 = nn.LeakyReLU(0.1)
        self.fc_c3 = nn.Linear(input_dim//4, lstm_size)
    
    def forward(self, x_a, x_p):
        lstm_out, lstm_h = self.lstm(x_a.view(len(x_a), -1, self.lstm_input_dim))
        hl_1 = self.fc_a1(lstm_out.view(len(x_a), -1, self.lstm_hidden_dim))
        hl_1 = self.lr1(hl_1)
        hl_2 = self.fc_a2(hl_1)
        hl_1 = self.lr1(hl_2)
        # hl_2 = self.fc_a2(hl_2)

        hl_p = self.fc_p1(x_p)
        hl_p = self.lr_p1(hl_p)

        # print('hidden_layer 1: ', hl_1.squeeze().shape)
        # print('hidden layer 2: ', hl_2.squeeze().shape)
        # print('hidden_layer p: ', hl_p.shape)
        
        combined = torch.cat((hl_2.view(hl_2.size(0), -1), hl_p), dim=1)
        # fullyconnected([10 timesteps] + [1 profile information])???

        # print('combined.shape: ', combined.shape)

        c_1 = self.fc_c1(combined)
        c_1 = self.lr3(c_1)
        c_2 = self.fc_c2(c_1)
        c_2 = self.lr4(c_2)
        c_3 = self.fc_c3(c_2)

        return c_3






if __name__ == "__main__":

    import os
    import sys
    sys.path.append(os.path.abspath(''))
    import util
    import pandas as pd
    from dataloader import dataset_non_ave_with_profile as dataset_class
    from torch.utils.data import DataLoader

    conditions = ['age']

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    lstm_size = 10
    step_size = 5
    params = {'batch_size': 16,
        'shuffle': True,
        'num_workers': 6}

    ## MODEL
    lstm_input_dim = 1582
    lstm_hidden_dim = 512
    fc_input_dim = len(conditions)
    fc_output_dim = 1
    model = Combination_model_2(lstm_input_dim, lstm_hidden_dim, fc_input_dim, fc_output_dim, lstm_size).to(device)
    model.float()
    print(model)
    # model.train()

    '''
    # load data
    '''
    feat_dict = util.load_pickle('data/feat_dict_ready.pkl')
    exps = pd.read_pickle(os.path.join('data', 'exps_ready.pkl'))
    pinfo = pd.read_pickle(os.path.join('data', 'pinfo_numero.pkl'))



    # prepare data for testing
    dataset_obj = dataset_class('arousals', feat_dict, exps, pinfo, conditions, lstm_size, step_size)
    dataset = dataset_obj.gen_dataset(train=False)
    loader = DataLoader(dataset, **params)

    for audio_info, profile_info, label in loader:
        audio_info, profile_info, label = audio_info.to(device).float(), profile_info.to(device).float(), label.to(device).float()
        print(audio_info.shape)
        print(profile_info.shape)
        print(label.shape)
        
        output = model.forward(audio_info,profile_info)
        print(output.shape)
        break
