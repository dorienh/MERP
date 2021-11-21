'''
since we are randomly selected segments of each song in each epoch, 
the number of epochs is explosive so by a high probability, every entire song is accounted for. lol.
'''

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
from torch.nn import functional as F
from torch.utils.data import DataLoader

import util
from util_method import save_model, pearson_corr_loss, load_model, average_exps_by_songurl

from testing_np_kfold import single_test, plot_pred_n_gts
### to edit accordingly.
from rdm_dataset import rdm_dataset as dataset_class

from networks import lstm_double as archi_lstm
from networks import Three_FC_layer as archi_linear


#####################
####    Train    ####
#####################
def train(train_loader, model, test_loader, fold_i, args):
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
            output = output.squeeze(1)
            # MSE Loss calculation
            loss_mse = F.mse_loss(output, label)
            loss_r = pearson_corr_loss(output, label)

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
        epoch_loss_log = {'mse' : [],'r' : []}

        for batchidx, (feature, label) in enumerate(train_loader):
            
            numbatches = len(train_loader)
            # Transfer to GPU
            feature, label = feature.to(device).float(), label.to(device).float()
            # clear gradients 
            optimizer.zero_grad()
            # forward pass
            output = model.forward(feature)
            output = output.squeeze(1)

            # loss
            loss_mse = F.mse_loss(output, label)
            loss_r = pearson_corr_loss(output, label)
            # loss = loss_mse*args.mse_weight + loss_r*args.r_weight

            # backward pass
            # loss.backward()
            loss_mse.backward()
            # update parameters
            optimizer.step()

            # record training loss
            epoch_loss_log['mse'].append(loss_mse.item())
            epoch_loss_log['r'].append(loss_r.item())

            print(f'Epoch: {epoch} || Batch: {batchidx}/{numbatches} || mse = {loss_mse.item():5f} || r = {loss_r.item():5f}', end = '\r')
            
            # val_loss += loss
        # log average loss
        aveloss_mse = np.average(epoch_loss_log['mse'])
        aveloss_r = np.average(epoch_loss_log['r'])
        # print(f'Initial round without training || mse = {aveloss_mse:.2f} || r = {aveloss_r:.2f}')
        loss_log['train_mse'].append(aveloss_mse)
        loss_log['train_r'].append(aveloss_r)
        print(' '*200)

        epoch_duration = time.time() - start_time
        print(f'Fold: {fold_i} || Epoch: {epoch:3} || mse: {aveloss_mse:8.5f} || r: {aveloss_r:8.5f} || time taken (s): {epoch_duration:8f}')
        

        test_ave_mse, test_ave_r = test(model, test_loader)
        print(f'test loss || mse: {test_ave_mse:.4f} || r: {test_ave_r:.4f}')
        loss_log['test_mse'].append(test_ave_mse)
        loss_log['test_r'].append(test_ave_r)

    # plot loss against epochs
    plt.plot(loss_log['train_mse'][1::], label='training loss mse')
    plt.plot(loss_log['test_mse'], label='test loss mse')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    # plt.ylim([0, 0.5])
    plt.title(f"Loss || init mse:{loss_log['train_mse'][0]:.3f} | test mse: {test_ave_mse:.3f}")
    plt.savefig(os.path.join(args.dir_path, 'saved_models', f'{args.model_name}', f'loss_plot_fold{fold_i}_mse.png'))
    plt.close()

    plt.plot(loss_log['train_r'][1::], label='training loss r')
    plt.plot(loss_log['test_r'], label='test loss r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    # plt.ylim([-1, 1])
    plt.legend()
    plt.title(f"Loss || init r:{loss_log['train_r'][0]:.3f} | test r: {test_ave_r:.3f}")
    plt.savefig(os.path.join(args.dir_path, 'saved_models', f'{args.model_name}', f'loss_plot_fold{fold_i}_r.png'))
    plt.close()

    return model, aveloss_mse, aveloss_r, test_ave_mse, test_ave_r

####################
####    Test    ####
####################

def test(model, test_loader):

    model.eval()
    losses = {'mse' : [], 'r' : []}
    with torch.no_grad():
        for feature, label in test_loader:
            feature, label = feature.to(device).float(), label.to(device).float()

            # forward pass
            output = model(feature)
            output = output.squeeze(1)

            # loss
            mse = F.mse_loss(output, label)
            r = pearson_corr_loss(output, label)

            losses['mse'].append(mse.item())
            losses['r'].append(r.item())
        
    return np.mean(losses['mse']), np.mean(losses['r'])




if __name__ == "__main__":
    
    ########################
    ####    Argparse    ####
    ########################

    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=dir_path)
    parser.add_argument('--linear', type=bool, default=False)
    parser.add_argument('--plot', type=bool, default=True)
    parser.add_argument('--master', type=int, default=1) # use int instead of boolean.
    parser.add_argument('--affect_type', type=str, default='valences', help='Can be either "arousals" or "valences"')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model_name', type=str, default='testlagi', help='Name of folder plots and model will be saved in')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_timesteps', type=int, default=30)
    # parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.0001)
    # parser.add_argument('--mse_weight', type=float, default=1.0)
    # parser.add_argument('--r_weight', type=float, default=1.0)
    # parser.add_argument('--conditions', nargs='+', type=str, default=[])

    args = parser.parse_args()
    if args.master == 0:
        save_models_foldername = 'saved_models'
    else:
        save_models_foldername = 'saved_models_m'

    if args.linear:
        setattr(args, 'model_name', f'linear_{args.affect_type[0]}_p_{args.model_name}')
        exp_log_filepath = os.path.join(dir_path,save_models_foldername,'test_log_linear.pkl')
        archi = archi_linear
    else:
        setattr(args, 'model_name', f'{args.affect_type[0]}_p_{args.model_name}')
        exp_log_filepath = os.path.join(dir_path,save_models_foldername,'test_log_lstm.pkl')
        archi = archi_lstm
    print(args)

    # check if folder with same model_name exists. if not, create folder.
    savepath = os.path.join(dir_path,save_models_foldername, args.model_name)
    os.makedirs(savepath, exist_ok=True)
    os.makedirs(os.path.join(savepath, 'predictions'), exist_ok=True)

    #########################
    ####    Load Data    ####
    #########################

    # read labels from pickle
    # exps = pd.read_pickle(f'data/exps_std_{args.affect_type[0]}_ave3.pkl')
    exps = pd.read_pickle('data/exps_ready3.pkl')

    if args.master == 1: # retrieve only master pinfos.
        pinfo = util.load_pickle('data/pinfo_numero.pkl')
        pinfo = pinfo[pinfo['master'] == 1.0]
        exps = exps[exps['workerid'].isin(pinfo['workerid'].unique())]
    
    # average the exps by song
    ave_exps = average_exps_by_songurl(exps, args.affect_type)
    exps = pd.DataFrame(list(zip(ave_exps.keys(),ave_exps.values())),  columns=['songurl', 'labels'])
    exps.set_index('songurl', inplace=True)
    
    
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

    ###########################
    ####    Dataloading    ####
    ###########################

    def dataloader_prep(feat_dict, exps, args, test=False):
        params = {
            'shuffle': True,
            'num_workers': args.num_workers,
            'batch_size': args.batch_size}

        if test:
            params['batch_size'] = 1
            seq_len = None
        else:
            seq_len=args.num_timesteps

        dataset = dataset_class(feat_dict, exps, seq_len=seq_len)
        loader = DataLoader(dataset, **params)
        return loader

    '''
    5 FOLD CROSS VALIDATION LEGGO
    '''
    loss_log_folds = {'train_loss_mse':[], 'train_loss_r':[], 'test_loss_mse':[], 'test_loss_r':[]}
    num_folds = 5
    for fold_i in range(num_folds):


        ########################
        ####    Training    ####
        ########################

        # load the data 
        # read audio features from pickle
        train_feat_dict = util.load_pickle(f'data/folds/train_feats_{fold_i}.pkl')
        test_feat_dict = util.load_pickle(f'data/folds/test_feats_{fold_i}.pkl')
        
        train_loader = dataloader_prep(train_feat_dict, exps, args, False)
        test_loader = dataloader_prep(test_feat_dict, exps, args, True)

        ###########################
        ####    Model param    ####
        ###########################

        ## MODEL
        input_dim = list(train_feat_dict.values())[0].shape[1] #724 # 1582 
        print('check input_dim: ', input_dim)
        model = archi(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
        model.float()
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        
        model, train_ave_mse, train_ave_r, test_ave_mse, test_ave_r = train(train_loader, model, test_loader, fold_i, args)

        save_model(model, savepath, f'{args.model_name}_{fold_i}')

        loss_log_folds['train_loss_mse'].append(train_ave_mse)
        loss_log_folds['train_loss_r']. append(train_ave_r)
        loss_log_folds['test_loss_mse'].append(test_ave_mse)
        loss_log_folds['test_loss_r']. append(test_ave_r)

        #######################
        ####    Testing    ####
        #######################

        # model = archi(input_dim).to(device)
        # model = load_model(model, savepath, f'{args.model_name}_{fold_i}')
        # test_ave_mse, test_ave_r, sum_test  = test(model, test_loader)

        if args.plot:
            for songurl in test_feat_dict.keys():
                _,_, pred_n_gts = single_test(model, device, songurl, test_feat_dict, exps)
                plot_pred_n_gts(pred_n_gts, songurl, args, filename_prefix=fold_i)

        # for songurl in util.trainlist[0:5]:
        #     single_test(model, songurl, train_feat_dict, exps, fold_i, args, 'train')
            
    # single_test(model, '0505_58', exps, args)

    # logging

    args_dict = vars(args)
    # print(type(args_dict))
    # args_dict['num_epochs'] = num_epochs
    ave_train_mse = np.mean(loss_log_folds["train_loss_mse"])
    ave_train_r = np.mean(loss_log_folds["train_loss_r"])
    args_dict['tr_mse'] = f'{ave_train_mse:.4f}'
    args_dict['tr_r'] = f'{ave_train_r:.4f}'
    # args_dict['tr_loss'] = f'{ave_train_mse+ave_train_r:.4f}'

    ave_test_mse = np.mean(loss_log_folds["test_loss_mse"])
    ave_test_r = np.mean(loss_log_folds["test_loss_r"])

    args_dict['tt_mse'] = f'{ave_test_mse:.4f}'
    args_dict['tt_r'] = f'{ave_test_r:.4f}'
    # args_dict['tt_loss'] = f'{ave_test_mse+ave_test_r:.4f}'
    for e in ['num_workers','dir_path', 'plot']:
        args_dict.pop(e)
    # print(args_dict)
    args_series = pd.Series(args_dict)
    args_df = args_series.to_frame().transpose()
    # print(args_df)

    if os.path.exists(exp_log_filepath):
        exp_log = pd.read_pickle(exp_log_filepath)
        exp_log = exp_log.append(args_df).reset_index(drop=True)
        pd.to_pickle(exp_log, exp_log_filepath)
        print(exp_log)
    else:
        pd.to_pickle(args_df, exp_log_filepath)
        print(args_df)