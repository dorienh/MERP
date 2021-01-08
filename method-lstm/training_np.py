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
from util_method import save_model, load_model, plot_pred_against, plot_pred_comparison, windowing, reverse_windowing
### to edit accordingly.
from dataloader import dataset_ave_no_profile as dataset_class
### to edit accordingly.
from network import LSTM_single as archi


#####################
####    Train    ####
#####################
def train(train_loader, model, test_loader, args):
    loss_log = {
        'train_mse' : [],
        'train_r' : [],
        'test_mse' : [],
        'test_r' : []
    }
    '''
        intial round
    '''
    with torch.no_grad():
        epoch_loss_log = {
            'mse' : [],
            'r' : []
        }
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
            loss_r = pearson_corr_loss(output.squeeze(), label.squeeze())

            epoch_loss_log['mse'].append(loss_mse.item())
            epoch_loss_log['r'].append(loss_r.item())
        
        aveloss_mse = np.average(epoch_loss_log['mse'])
        aveloss_r = np.average(epoch_loss_log['r'])
        print(f'Initial round without training || mse = {aveloss_mse:.2f} || r = {aveloss_r:.2f}')
        loss_log['train_mse'].append(aveloss_mse)
        loss_log['train_r'].append(aveloss_r)
    
    '''
        actual training
    '''
    for epoch in np.arange(args.num_epochs):
        model.train()
        start_time = time.time()
        epoch_loss_log = {
            'mse' : [],
            'r' : []
        }

        for batchidx, (feature, label) in enumerate(train_loader):
            
            numbatches = len(train_loader)
            # Transfer to GPU
            feature, label = feature.to(device).float(), label.to(device).float()
            # clear gradients 
            optimizer.zero_grad()
            # forward pass
            output = model.forward(feature)

            # loss
            loss_mse = nn.MSELoss()(output.squeeze(), label.squeeze())
            loss_r = pearson_corr_loss(output.squeeze(), label.squeeze())
            loss = loss_mse + loss_r

            # backward pass
            loss.backward(retain_graph=True)
            # update parameters
            optimizer.step()

            # record training loss
            epoch_loss_log['mse'].append(loss_mse.item())
            epoch_loss_log['r'].append(loss_r.item())

            print(f'Epoch: {epoch} || Batch: {batchidx}/{numbatches} || mse = {loss_mse.item():5f} || r = {loss_r.item():5f}', end = '\r')

        # log average loss
        aveloss_mse = np.average(epoch_loss_log['mse'])
        aveloss_r = np.average(epoch_loss_log['r'])
        # print(f'Initial round without training || mse = {aveloss_mse:.2f} || r = {aveloss_r:.2f}')
        loss_log['train_mse'].append(aveloss_mse)
        loss_log['train_r'].append(aveloss_r)
        print(' '*200)

        epoch_duration = time.time() - start_time
        print(f'Epoch: {epoch:3} || mse: {aveloss_mse:8.5f} || r: {aveloss_r:8.5f} || time taken (s): {epoch_duration:8f}')
        
        test_ave_mse, test_ave_r  = test(model, test_loader)
        loss_log['test_mse'].append(test_ave_mse)
        loss_log['test_r'].append(test_ave_r)

    # plot loss against epochs
    plt.plot(loss_log['train_mse'][1::], label='training loss mse')
    plt.plot(loss_log['test_mse'], label='test loss mse')
    plt.plot(loss_log['train_r'][1::], label='training loss r')
    plt.plot(loss_log['test_r'], label='test loss r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim([-1, 1])
    plt.legend()
    plt.title(f"Loss || init mse:{loss_log['train_mse'][0]:.3f} | r:{loss_log['train_r'][0]:.3f} || test mse: {loss_log['test_mse'][-1]:.3f} | r: {loss_log['test_r'][-1]:.3f}")
    plt.savefig(os.path.join(args.dir_path, 'saved_models', f'{args.model_name}', 'loss_plot.png'))
    plt.close()

    return model, test_ave_mse, test_ave_r

####################
####    Test    ####
####################

def test(model, test_loader):

    model.eval()

    epoch_loss_log = {
            'mse' : [],
            'r' : []
        }
    with torch.no_grad():
        # model = load_model(model_name)
        for batchidx, (feature, label) in enumerate(test_loader):
            
            feature, label = feature.to(device).float(), label.to(device).float()
            # forward pass
            output = model(feature)
            # MSE Loss calculation

            loss_mse = nn.MSELoss()(output.squeeze(), label.squeeze())
            loss_r = pearson_corr_loss(output.squeeze(), label.squeeze())
            # loss = loss_mse + loss_r
            # print(loss)
            epoch_loss_log['mse'].append(loss_mse.item())
            epoch_loss_log['r'].append(loss_r.item())
    
    test_ave_mse = np.average(epoch_loss_log['mse'])
    test_ave_r = np.average(epoch_loss_log['r'])
    print(f'average test lost (per batch): mse: {test_ave_mse:4f} r: {test_ave_r:4f}')
    return test_ave_mse, test_ave_r


def single_test(model, index, songurl, exps, args):
    '''
        exps - the original exps with many workers
    '''
    # features - audio
    testfeat = test_feat_dict[songurl]
    # features - exps
    # testtrial = exps[exps['songurl']==songurl].reset_index().loc[index]
    testlabel = exps.at[songurl,'labels']
    # labels
    # testlabel = testtrial[args.affect_type]

    testinput = testfeat

    print('testinput ', testinput.shape)
    print('testlabel ', testlabel.shape)

    with torch.no_grad():
        testinput = torch.from_numpy(testinput)
        testlabel = torch.from_numpy(testlabel)

        feature, label = testinput.to(device).float(), testlabel.to(device).float()

        # forward pass
        output = model(feature)
        # MSE Loss calculation
        # loss = criterion(output.squeeze(), label.squeeze())
        loss_mse = nn.MSELoss()(output.squeeze(), label.squeeze())
        loss_r = pearson_corr_loss(output.squeeze(), label.squeeze())
        # loss = loss_mse + loss_r
    # print(loss.item())

    output = reverse_windowing(output, args.lstm_size, args.step_size)
    print('meow: ', output.shape)
    label = reverse_windowing(label, args.lstm_size, args.step_size)
    print('label: ', label.shape)

    dir_path = os.path.dirname(os.path.realpath(__file__))

    plt = plot_pred_comparison(output, label, loss_mse.item(), loss_r.item())
    plt.suptitle(f'{songurl}')
    plt.savefig(os.path.join(dir_path, 'saved_models', f'{args.model_name}/predictions/{songurl}_prediction_{index}.png'))
    plt.close()

    plt = plot_pred_against(output, label)
    plt.suptitle(f'{songurl}')
    plt.savefig(os.path.join(dir_path, 'saved_models', f'{args.model_name}/predictions/{songurl}_y_vs_yhat_{index}.png'))
    plt.close()



####################
####    Misc    ####
####################

def dataloader_prep(feat_dict, exps, args, train=True):
    params = {'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers}
    # prepare data for testing
    dataset_obj = dataset_class(feat_dict, exps, train)
    dataset = dataset_obj.gen_dataset()#train=train)
    loader = DataLoader(dataset, **params)

    return loader

if __name__ == "__main__":
    
    ########################
    ####    Argparse    ####
    ########################

    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=dir_path)
    parser.add_argument('--affect_type', type=str, default='arousals', help='Can be either "arousals" or "valences"')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='testing100epochs', help='Name of folder plots and model will be saved in')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--drop_prob', type=float, default=0.1)
    parser.add_argument('--lstm_size', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    # parser.add_argument('--conditions', nargs='+', type=str, default=[])

    args = parser.parse_args()
    setattr(args, 'model_name', f'{args.affect_type[0]}_np_{args.model_name}')
    print(args)

    # check if folder with same model_name exists. if not, create folder.
    os.makedirs(os.path.join(dir_path,'saved_models', args.model_name), exist_ok=True)
    os.makedirs(os.path.join(dir_path,'saved_models', args.model_name, 'predictions'), exist_ok=True)

    #########################
    ####    Load Data    ####
    #########################

    # load the data 
    # read audio features from pickle
    train_feat_dict = util.load_pickle('data/train_feats_pca_windowed.pkl')
    test_feat_dict = util.load_pickle('data/test_feats_pca_windowed.pkl')
    # read labels from pickle
    exps = pd.read_pickle('data/exps_std_a_ave_windowed.pkl')
    original_exps = pd.read_pickle('data/exps_ready.pkl')
    
    ####################
    ####    Cuda    ####
    ####################

    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    print('cuda: ', use_cuda)
    print('device: ', device)

    ####################
    ####    Loss    ####
    ####################

    def pearson_corr_loss(output, target):
        x = output
        y = target

        vx = x - torch.mean(x)
        vy = y - torch.mean(y)

        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
        return cost*-1

    ###########################
    ####    Model param    ####
    ###########################

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

    # save_model(model, args.model_name, dir_path)

    #######################
    ####    Testing    ####
    #######################

    # model = archi(input_dim).to(device)
    # model = load_model(model, args.model_name, dir_path)

    for songurl in util.testlist:
        # single_test(model, 1, songurl, original_exps, args)
        single_test(model, 1, songurl, exps, args)

    # # logging

    # args_dict = vars(args)
    # # print(type(args_dict))
    # args_dict['test_loss_mse'] = f'{test_ave_mse:.6f}'
    # args_dict['test_loss_r'] = f'{test_ave_r:.6f}'
    # args_dict['test_loss'] = f'{test_ave_mse+test_ave_r:.6f}'
    # args_dict.pop('dir_path')
    # # print(args_dict)
    # args_series = pd.Series(args_dict)
    # args_df = args_series.to_frame().transpose()
    # # print(args_df)

    # exp_log_filepath = os.path.join(dir_path,'saved_models','experiment_log.pkl')
    # if os.path.exists(exp_log_filepath):
    #     exp_log = pd.read_pickle(exp_log_filepath)
    #     exp_log = exp_log.append(args_df).reset_index(drop=True)
    #     pd.to_pickle(exp_log, exp_log_filepath)
    #     print(exp_log)
    # else:
    #     pd.to_pickle(args_df, exp_log_filepath)
    #     print(args_df)