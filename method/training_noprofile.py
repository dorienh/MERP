import os
import sys
# sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath(''))
print(sys.path)
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
from util_method import save_model, load_model, plot_pred_against, plot_pred_comparison, standardize
### to edit accordingly.
from dataloader import dataset_ave_no_profile as dataset_class
### to edit accordingly.
from network import Two_FC_layer as archi


def dataloader_prep(feat_dict, exps, args, train=True):
    params = {'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers}
    # prepare data for testing
    dataset_obj = dataset_class(args.affect_type, feat_dict, exps, train)
    dataset = dataset_obj.gen_dataset()#train=train)
    loader = DataLoader(dataset, **params)

    return loader



def train(train_loader, model, test_loader, args):
    loss_epoch_log = []
    test_loss_epoch_log = []

    with torch.no_grad():
        loss_log = []
        for batchidx, (feature, label) in enumerate(train_loader):
            numbatches = len(train_loader)
            # Transfer to GPU
            feature, label = feature.to(device).float(), label.to(device).float()
            # clear gradients 
            optimizer.zero_grad()
            # forward pass
            output = model.forward(feature)
            # MSE Loss calculation
            loss_mse = nn.MSELoss()(output.squeeze(), label.squeeze())
            loss_r = criterion(output.squeeze(), label.squeeze())
            loss = loss_mse + loss_r
            loss_log.append(loss.item())
        aveloss = np.average(loss_log)
        loss_epoch_log.append(aveloss)
        print(f'Initial round without training || MSELoss = {aveloss:.6f}')
    
    for epoch in np.arange(args.num_epochs):
        model.train()
        start_time = time.time()
        loss_log = []    

        # Training
        for batchidx, (feature, label) in enumerate(train_loader):
            numbatches = len(train_loader)
            # Transfer to GPU
            feature, label = feature.to(device).float(), label.to(device).float()
            # clear gradients 
            optimizer.zero_grad()
            # forward pass
            output = model.forward(feature)
            
            # get and record accuracy
            # print(output.data)
            _, pred = torch.max(output.data, 1)
            accuracy_sum = torch.sum(pred == label)
            accuracy = np.float32(accuracy_sum.item()/output.size()[0])
            # accuracy_log.append(accuracy)

            # MSE Loss calculation
            # loss = criterion(output.squeeze(), label.squeeze())
            loss_mse = nn.MSELoss()(output.squeeze(), label.squeeze())
            loss_r = criterion(output.squeeze(), label.squeeze())
            loss = loss_mse + loss_r
            # backward pass
            loss.backward(retain_graph=True)
            # update parameters
            optimizer.step()
            # record training loss
            loss_log.append(loss.item())
            print(f'Epoch: {epoch} || Batch: {batchidx}/{numbatches} || Loss = {loss.item()}', end = '\r')
            

        aveloss = np.average(loss_log)
        print(' '*200)
        loss_epoch_log.append(aveloss)
        
        epoch_duration = time.time() - start_time
        print(f'Epoch: {epoch:3} || MSELoss: {aveloss:10.6f} || time taken (s): {epoch_duration}')
    
        test_loss_epoch_log.append(test(model, test_loader))

    # plot loss against epochs
    plt.plot(loss_epoch_log[1::], label='training loss')
    plt.plot(test_loss_epoch_log, label='test loss')
    plt.xlabel('epoch')
    plt.ylabel('mseloss')
    plt.legend()
    plt.title(f'Loss || before training: {loss_epoch_log[0]:.6f} || test loss: {test_loss_epoch_log[-1]:.6f}')
    plt.savefig(os.path.join(args.dir_path, 'saved_models', f'{args.model_name}', 'loss_plot.png'))
    plt.close()

    return model, test_loss_epoch_log[-1]


def single_test(model, index, args):
    # features - audio
    testfeat = test_feat_dict['00_145']
    # features - pinfo
    testtrial = exps[exps['songurl']=='00_145'].reset_index().loc[index]
    # labels
    testlabel = testtrial[args.affect_type]

    testinput = testfeat

    with torch.no_grad():
        testinput = torch.from_numpy(testinput)
        testlabel = torch.from_numpy(testlabel)

        feature, label = testinput.to(device).float(), testlabel.to(device).float()
        model = load_model(model, args.model_name, args.dir_path)

        # forward pass
        output = model(feature)
        # MSE Loss calculation
        # loss = criterion(output.squeeze(), label.squeeze())
        loss_mse = nn.MSELoss()(output.squeeze(), label.squeeze())
        loss_r = criterion(output.squeeze(), label.squeeze())
        loss = loss_mse + loss_r
    # print(loss.item())

    dir_path = os.path.dirname(os.path.realpath(__file__))

    plt = plot_pred_comparison(output, label, loss.item())
    plt.savefig(os.path.join(dir_path, 'saved_models', f'{args.model_name}/prediction_{index}.png'))
    plt.close()

    plt = plot_pred_against(output, label, loss.item())
    plt.savefig(os.path.join(dir_path, 'saved_models', f'{args.model_name}/y_vs_yhat_{index}.png'))
    plt.close()

def test(model, test_loader):

    model.eval()

    loss_log = []
    with torch.no_grad():
        # model = load_model(model_name)
        for batchidx, (feature, label) in enumerate(test_loader):
            
            feature, label = feature.to(device).float(), label.to(device).float()
            # forward pass
            output = model(feature)
            # MSE Loss calculation
            # loss = criterion(output.squeeze(), label.squeeze())
            loss_mse = nn.MSELoss()(output.squeeze(), label.squeeze())
            loss_r = criterion(output.squeeze(), label.squeeze())
            loss = loss_mse + loss_r
            # print(loss)
            loss_log.append(loss.item())
    aveloss = np.average(loss_log)
    print(f'average test lost (per batch): {aveloss}')
    return aveloss



if __name__ == "__main__":

    # current file path.
    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=dir_path)

    parser.add_argument('--affect_type', type=str, default='arousals', help='Can be either "arousals" or "valences"')
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--model_name', type=str, default='ave_customloss3', help='Name of folder plots and model will be saved in')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lstm_size', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--conditions', nargs='+', type=str, default=[])


    args = parser.parse_args()
    setattr(args, 'model_name', f'{args.affect_type[0]}_np_{args.model_name}')
    print(args)

    # check if folder with same model_name exists. if not, create folder.
    os.makedirs(os.path.join(dir_path,'saved_models', args.model_name), exist_ok=True)
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    print('cuda: ', use_cuda)
    print('device: ', device)

    '''
    load data
    '''
    feat_dict = util.load_pickle('data/feat_dict_ready.pkl')
    train_feat_dict = util.load_pickle('data/train_feats_pca.pkl')
    test_feat_dict = util.load_pickle('data/test_feats_pca.pkl')
    # print(train_feat_dict.keys())
    exps = pd.read_pickle(os.path.join('data', 'exps_ready.pkl'))
    # pinfo = pd.read_pickle(os.path.join('data', 'pinfo_numero.pkl'))

    # standardize audio features
    # feat_dict = standardize(feat_dict)
    # train

    ## MODEL
    input_dim = 724 # 1582 
    model = archi(input_dim=input_dim).to(device)
    model.float()
    print(model)


    def my_loss(output, target):
        loss = torch.mean((output - target)**2)
        return loss
    
    def pearson_corr_loss(output, target):
        x = output
        y = target

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost**2 

    criterion = pearson_corr_loss # nn.MSELoss()

    

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = dataloader_prep(train_feat_dict, exps, args, train=True)
    test_loader = dataloader_prep(test_feat_dict, exps, args, train=False)
    
    model, testloss = train(train_loader, model, test_loader, args)
    save_model(model, args.model_name, dir_path)

    model = archi(input_dim).to(device)
    model = load_model(model, args.model_name, dir_path)
    single_test(model, 1, args)

    ## logging

    args_dict = vars(args)
    # print(type(args_dict))
    args_dict['test_loss'] = f'{testloss:.6f}'
    args_dict.pop('dir_path')
    # print(args_dict)
    args_series = pd.Series(args_dict)
    args_df = args_series.to_frame().transpose()
    # print(args_df)

    exp_log_filepath = os.path.join(dir_path,'saved_models','experiment_log.pkl')
    if os.path.exists(exp_log_filepath):
        exp_log = pd.read_pickle(exp_log_filepath)
        exp_log = exp_log.append(args_df).reset_index(drop=True)
        pd.to_pickle(exp_log, exp_log_filepath)
        print(exp_log)
    else:
        pd.to_pickle(args_df, exp_log_filepath)
        print(args_df)