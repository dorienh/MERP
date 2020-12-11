import os
import sys
sys.path.append(os.path.abspath(''))
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse
import copy
# from sklearn.model_selection import KFold 

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import util
from util_method import save_model, load_model, plot_pred_against, plot_pred_comparison, standardize, combine_similar_pinfo
### to edit accordingly.
from dataloader import dataset_non_ave_with_profile as dataset_class
### to edit accordingly.
from network import Combination_model_2 as archi

def make_folds(exps, number_of_folds):
    '''
    all 10 folds have deam songs, but the other FMA songs do not overlap in folds. 
    this may make the folds less similar.
    '''

    # kf = KFold(n_splits=10, random_state=42, shuffle=True)

    deam_folds_list = []

    for deamsong in util.songdict['deam']:
        trials = exps.loc[exps['songurl'] == deamsong]
        trials_folds = np.array_split(trials, number_of_folds)
        deam_folds_list.append(trials_folds)

    folds = []

    for i, songurl_fold in util.folds.items():
        # print(i)
        trials = pd.DataFrame()
        for songurl in songurl_fold:
            trials = trials.append(exps.loc[exps['songurl'] == songurl])
            # print(len(trials))
        for deam_fold in deam_folds_list:
            # print(len(deam_fold[i]))
            trials = trials.append(deam_fold[i])
            # print(len(trials))
        
        folds.append(trials)

    return folds

def train_test_by_fold(folds, test_fold_index):
    '''
    returns the train and test dataframes given the index of the fold that is to be the test set.
    '''
    # test
    test_df = folds.pop(test_fold_index)
    # train
    train_df = pd.DataFrame()
    for fold in folds:
        train_df = train_df.append(fold)
    

    return train_df, test_df

def dataloader_prep(feat_dict, exps, pinfo, args):
    params = {'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers}
    # prepare data for testing
    dataset_obj = dataset_class(args.affect_type, feat_dict, exps, pinfo, args.conditions, args.lstm_size, args.step_size)
    dataset = dataset_obj.gen_dataset()
    loader = DataLoader(dataset, **params)

    return loader



def train(train_loader, model, test_loader, fold_index, args):

    model.train()
    loss_epoch_log = []
    test_loss_epoch_log = []

    # one round without training.
    loss_log = []
    for batchidx, (audio_info, profile_info, label) in enumerate(train_loader):
        numbatches = len(train_loader)
        label = label[:,-1] # we want to train the model to predict the last timestep.
        # print(label)
        # Transfer to GPU
        audio_info, profile_info, label = audio_info.to(device).float(), profile_info.to(device).float(), label.to(device).float()
        # print('audio info shape: ', audio_info.shape)
        # print('profile info shape: ', profile_info.shape)
        # print('label shape: ', label.shape)

        # clear gradients 
        optimizer.zero_grad()
        # forward pass
        output = model.forward(audio_info, profile_info)
        # MSE Loss calculation
        loss = criterion(output.squeeze(), label.squeeze())
        loss_log.append(loss.item())
    aveloss = np.average(loss_log)
    loss_epoch_log.append(aveloss)
    print(f'Initial round without training || MSELoss = {aveloss:.6f}')

    for epoch in np.arange(1, args.num_epochs+1):
        model.train()
        start_time = time.time()
        loss_log = []

        # Training
        for batchidx, (audio_info, profile_info, label) in enumerate(train_loader):
            numbatches = len(train_loader)
            label = label[:,-1] # we want to train the model to predict the last timestep.
            # print(label)
            # Transfer to GPU
            audio_info, profile_info, label = audio_info.to(device).float(), profile_info.to(device).float(), label.to(device).float()
            
            # clear gradients 
            model.zero_grad()
            # optimizer.zero_grad()
            # forward pass
            output = model.forward(audio_info, profile_info)
            # MSE Loss calculation
            loss = criterion(output.squeeze(), label.squeeze())
            # backward pass
            loss.backward(retain_graph=True)
            # update parameters
            optimizer.step()
            # record training loss
            loss_log.append(loss.item())
            print(f'Epoch: {epoch} || Batch: {batchidx}/{numbatches} || MSELoss = {loss.item()}', end = '\r')
            
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
    plt.savefig(os.path.join(args.dir_path, 'saved_models', 'loss_plots', f'{args.model_name}_loss_plot_{fold_index}.png'))
    plt.close()

    return model, loss_epoch_log, test_loss_epoch_log

def test(model, test_loader):

    model.eval()

    loss_log = []
    with torch.no_grad():

        for batchidx, (audio_info, profile_info, label) in enumerate(test_loader):
            label = label[:,-1] # we want to train the model to predict the last timestep.
            audio_info, profile_info, label = audio_info.to(device).float(), profile_info.to(device).float(), label.to(device).float()
            # forward pass
            output = model(audio_info, profile_info)
            # MSE Loss calculation
            loss = criterion(output.squeeze(), label.squeeze())
            # print(loss)
            loss_log.append(loss.item())
    aveloss = np.average(loss_log)
    print(f'average test lost (per batch): {aveloss} \n')
    return aveloss



if __name__ == "__main__":

    # current file path.
    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=dir_path)

    parser.add_argument('--affect_type', type=str, default='arousals', help='Can be either "arousals" or "valences"')
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='median_model_2_test', help='Name of folder plots and model will be saved in')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lstm_hidden_dim', type=int, default=512)
    parser.add_argument('--lstm_size', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--drop_prob', type=int, default=0.01)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--number_of_folds', type=int, default=10)
    parser.add_argument('--conditions', nargs='+', type=str, default=['age'])

    parser.add_argument('--mean', type=bool, default=False)
    parser.add_argument('--median', type=bool, default=False)

    args = parser.parse_args()
    setattr(args, 'model_name', f'{args.affect_type[0]}_p_{args.model_name}')
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

    # possible_conditions = np.array(['age', 'country_enculturation', 'country_live', 'fav_music_lang', 'gender', 'fav_genre', 'play_instrument', 'training', 'training_duration'])
    '''
    # load data
    '''
    feat_dict = util.load_pickle('data/feat_dict_ready.pkl')
    exps = pd.read_pickle(os.path.join('data', 'exps_ready.pkl'))
    pinfo = pd.read_pickle(os.path.join('data', 'pinfo_numero.pkl'))

    # standardize audio features
    feat_dict = standardize(feat_dict)

    if args.conditions and (args.mean != False or args.median != False):
        exps = combine_similar_pinfo(pinfo, exps, args)
    print(exps.shape)

    folds = make_folds(exps, args.number_of_folds)
    
    ## MODEL
    lstm_input_dim = 1582
    fc_input_dim = len(args.conditions)
    fc_output_dim = fc_input_dim if fc_input_dim==1 else fc_input_dim//2
    model = archi(lstm_input_dim, args.lstm_hidden_dim, fc_input_dim, fc_output_dim, args.lstm_size, args.drop_prob).to(device)
    model.float()
    print(model)
    model.train()
    

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    
    # train each fold, storing the losses every epoch
    for fold_index in range(args.number_of_folds):
        
        print(f'training fold number {fold_index}')

        folds_copy = copy.deepcopy(folds)
        train_df, test_df = train_test_by_fold(folds_copy, fold_index)
    

        train_loader = dataloader_prep(feat_dict, train_df, pinfo, args)
        test_loader = dataloader_prep(feat_dict, test_df, pinfo, args)
        model, trainloss, testloss = train(train_loader, model, test_loader, fold_index, args)
    
    # save_model(model, args.model_name, dir_path)
    
    # model = archi(lstm_input_dim, args.lstm_hidden_dim, fc_input_dim, fc_output_dim, args.lstm_size).to(device)
    # model = load_model(model, args.model_name, dir_path)
    # single_test(model, 1, args)

    # # test_loader = dataloader_prep(feat_dict, exps, pinfo, args, train=False)
    # # testloss = test(model, test_loader)

    # # plot losses with error bars (barplot with error bars against epochs on x axis)
    
    # #return testloss
    # # args_namespace = argparse.Namespace()
    # args_dict = vars(args)
    # # print(type(args_dict))
    # args_dict['test_loss'] = f'{testloss:.6f}'
    # args_dict['train_loss'] = f'{trainloss:.6f}'
    # args_dict.pop('dir_path')
    # # print(args_dict)
    # args_series = pd.Series(args_dict)
    # args_df = args_series.to_frame().transpose()
    # # print(args_df)

    # # exp_log_filepath = os.path.join(dir_path,'saved_models','experiment_log4.pkl')
    # # if os.path.exists(exp_log_filepath):
    # #     exp_log = pd.read_pickle(exp_log_filepath)
    # #     exp_log = exp_log.append(args_df).reset_index(drop=True)
    # #     pd.to_pickle(exp_log, exp_log_filepath)
    # #     print(exp_log)
    # # else:
    # #     pd.to_pickle(args_df, exp_log_filepath)
    # #     print(args_df)
    