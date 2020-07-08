import torch
from torch import nn
from torch.nn import functional as F
import torch



##############################################################
####                  1) Fully Connected                  ####
##############################################################

class Mult_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 1582, reduced_dim_power_2=9):
        super(Mult_FC_layer, self).__init__()
        # add more gradual change in dim. 

        self.hidden = nn.ModuleList()

        self.hidden.append(nn.Linear(input_dim, 2**reduced_dim_power_2, bias=False))

        power_2 = reduced_dim_power_2 - 1
        while power_2 >= 4:
            self.hidden.append(nn.Linear(2**(power_2+1), 2**power_2, bias=False))
            self.hidden.append(nn.LeakyReLU(0.1))
            power_2 -= 1

        self.hidden.append(nn.Linear(2**(power_2+1), out_features=1, bias=False)) # output

    def forward(self, x):
        for i, l in enumerate(self.hidden):
            # x = self.hidden[i // 2](x) + l(x)
            x = l(x)
        return x


##############################################################
####            2) Convolutional Neural Network           ####
##############################################################


class Simple_CNN_Reg(torch.nn.Module):
    def __init__(self, input_dim = 1582):
        super(Simple_CNN_Reg, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim,out_channels=128,kernel_size=(5))

    def forward(self, x):
        x = self.conv1(x)
        
        return x

##############################################################
####                        2) LSTM                       ####
##############################################################

class LSTM_single(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(LSTM_single, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.act1 = nn.Tanh()

        self.fc = nn.Linear(hidden_dim, hidden_dim//2)
        self.lr2 = nn.LeakyReLU(0.1)
        
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.lr3 = nn.LeakyReLU(0.1)

        self.fc3 = nn.Linear(hidden_dim//4, 1)
        self.input_dim = input_dim

    def forward(self, x):
        # print(x.view(len(x), -1, self.input_dim).shape)
        lstm_out, lstm_h = self.lstm(x.view(len(x), -1, self.input_dim))
        lstm_out = self.act1(lstm_out)
        prediction = self.lr2(self.fc(lstm_out.view(len(x),-1, self.hidden_dim)))
        prediction = self.lr3(self.fc2(prediction))
        prediction = self.fc3(prediction)

        return prediction



##############################################################
####               2) LSTM and FC profile                 ####
##############################################################

class Combination_model(torch.nn.Module):
    def __init__(self, lstm_input_dim, lstm_hidden_dim, fc_input_dim, fc_output_dim):
        super(Combination_model, self).__init__()
        self.lstm_input_dim = lstm_input_dim
        self.lstm_hidden_dim = lstm_hidden_dim

        self.lstm = nn.LSTM(lstm_input_dim, lstm_hidden_dim)

        self.fc_a1 = nn.Linear(lstm_hidden_dim, lstm_hidden_dim//2)
        self.lr1 = nn.LeakyReLU(0.1)
        self.fc_a2 = nn.Linear(lstm_hidden_dim//2, lstm_hidden_dim//4)
        self.lr2 = nn.LeakyReLU(0.1)
        self.fc_a3 = nn.Linear(lstm_hidden_dim//4, 1)

        self.fc_p1 = nn.Linear(fc_input_dim, fc_output_dim)
    
    def forward(self, x_a, x_p):
        lstm_out, lstm_h = self.lstm(x_a) #_a.view(len(x_a), -1, self.lstm_input_dim))

        return lstm_out


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
    
    

    ## MODEL
    lstm_input_dim = 1582 + len(conditions)
    lstm_hidden_dim = 512
    fc_input_dim = len(conditions)
    fc_output_dim = 1
    model = LSTM_single(lstm_input_dim, lstm_hidden_dim).to(device)
    model.float()
    print(model)
    # model.train()

    '''
    # load data
    '''
    feat_dict = util.load_pickle('data/feat_dict_ready.pkl')
    exps = pd.read_pickle(os.path.join('data', 'exps_ready.pkl'))
    pinfo = pd.read_pickle(os.path.join('data', 'pinfo_numero.pkl'))


    lstm_size = 10
    step_size = 5
    params = {'batch_size': 16,
        'shuffle': True,
        'num_workers': 6}
    # prepare data for testing
    dataset_obj = dataset_class('arousals', feat_dict, exps, pinfo, conditions, lstm_size, step_size)
    dataset = dataset_obj.gen_dataset(train=False)
    loader = DataLoader(dataset, **params)

    for feature, label in loader:
        feature, label = feature.to(device).float(), label.to(device).float()
        print(feature.shape)
        print(label.shape)
        
        output = model.forward(feature)
        print(output.shape)
        break

    


'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
'''









