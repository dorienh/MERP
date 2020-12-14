import torch
from torch import nn
from torch.nn import functional as F
import torch



##############################################################
####                  1) Fully Connected                  ####
##############################################################

class Two_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 724, reduced_dim=128, fc_dim = 64):
    # def __init__(self, input_dim = 1582, reduced_dim=128, fc_dim = 64):
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

class Mult_FC_layer(torch.nn.Module):
    def __init__(self, input_dim = 724, reduced_dim_power_2=9):
    # def __init__(self, input_dim = 1582, reduced_dim_power_2=9):
        super(Mult_FC_layer, self).__init__()
        # add more gradual change in dim. 

        self.hidden = nn.ModuleList()

        self.hidden.append(nn.Linear(input_dim, 2**reduced_dim_power_2, bias=False))

        power_2 = reduced_dim_power_2 - 1
        while power_2 >= 4:
            self.hidden.append(nn.Linear(2**(power_2+1), 2**power_2, bias=False))
            self.hidden.append(nn.LeakyReLU(0.1))
            self.hidden.append(nn.Dropout(p=0.5))
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
        None

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









