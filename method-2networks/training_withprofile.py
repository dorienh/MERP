import os
import sys
# sys.path.append(os.path.abspath('../..'))
# sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath(''))
# print(sys.path)
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import argparse

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

import util
from util_method import save_model, load_model, plot_pred_against, plot_pred_comparison, standardize
### to edit accordingly.
from dataloader import dataset_non_ave_with_profile as dataset_class
### to edit accordingly.
from network import Combination_model_1 as archi


def dataloader_prep(feat_dict, exps, pinfo, args, train=True):
    params = {'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers}
    # prepare data for testing
    dataset_obj = dataset_class(args.affect_type, feat_dict, exps, pinfo, args.conditions, args.lstm_size, args.step_size)
    dataset = dataset_obj.gen_dataset(train=train)
    loader = DataLoader(dataset, **params)

    return loader


def train(train_loader, model, test_loader, args):
    model.train()
    loss_epoch_log = []
    test_loss_epoch_log = []

    # one round without training.
    loss_log = []
    for batchidx, (audio_info, profile_info, label) in enumerate(train_loader):
        numbatches = len(train_loader)
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
    plt.savefig(os.path.join(args.dir_path, 'saved_models', f'{args.model_name}', 'loss_plot.png'))
    plt.close()

    return model, test_loss_epoch_log[-1]

def single_test(model, index, args): ## doesn't work... not sure how to reverse unfold yet.
    # # features - audio
    audio_feat = torch.from_numpy(np.array(feat_dict['00_145']))
    w_audio_feat = audiofeat.unfold(0,args.lstm_size, args.step_size)


    w_label = label.unfold(0,lstm_size, step)

    # # features - pinfo
    test_exp = exps[exps['songurl']=='00_145'].reset_index().loc[index]
    test_exp = pd.DataFrame(test_exp).transpose()
    print(test_exp)
    # workerid = test_exp['workerid']
    # workerinfo = pinfo.loc[pinfo.workerid.str.contains(workerid)][args.conditions]
    # workerinfo = workerinfo.values.tolist()[0]
    params = {'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers}
    
    
    # print(workerinfo)
    
    # labels
    ground_truth = test_exp[args.affect_type]

    output_list = []

    with torch.no_grad():

        for batchidx, (audio_info, profile_info, label) in enumerate(single_test_dl):
            
            audio_info, profile_info, label = audio_info.to(device).float(), profile_info.to(device).float(), label.to(device).float()
            # forward pass
            output = model(audio_info, profile_info)
            # MSE Loss calculation
            loss = criterion(output.squeeze(), label.squeeze())
            # print(loss)
            loss_log.append(loss.item())
            
            output_list.append(output.squeeze())
    
    prediction = [output_list[0]]
    for i in range(1, len(output_list)):
        prediction.append(output_list[args.stepsize::])

    print(prediction)
    

    plt = plot_pred_comparison(output.squeeze(), label, loss.item())
    plt.savefig(os.path.join(args.dir_path, 'saved_models', f'{args.model_name}', f'prediction_{index}.png'))
    plt.close()

    plt = plot_pred_against(output.squeeze(), label, loss.item())
    plt.savefig(os.path.join(args.dir_path, 'saved_models', f'{args.model_name}', f'y_vs_yhat_{index}.png'))
    plt.close()

def test(model, test_loader):

    model.eval()

    loss_log = []
    with torch.no_grad():

        for batchidx, (audio_info, profile_info, label) in enumerate(test_loader):
            
            audio_info, profile_info, label = audio_info.to(device).float(), profile_info.to(device).float(), label.to(device).float()
            # forward pass
            output = model(audio_info, profile_info)
            # MSE Loss calculation
            loss = criterion(output.squeeze(), label.squeeze())
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
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--model_name', type=str, default='condition_tr_model_1', help='Name of folder plots and model will be saved in')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--lstm_hidden_dim', type=int, default=512)
    parser.add_argument('--lstm_size', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--conditions', nargs='+', type=str, default=['training'])

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

    ## MODEL
    lstm_input_dim = 1582
    fc_input_dim = len(args.conditions)
    fc_output_dim = fc_input_dim if fc_input_dim==1 else fc_input_dim//2
    model = archi(lstm_input_dim, args.lstm_hidden_dim, fc_input_dim, fc_output_dim, args.lstm_size).to(device)
    model.float()
    print(model)
    model.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_loader = dataloader_prep(feat_dict, exps, pinfo, args, train=True)
    test_loader = dataloader_prep(feat_dict, exps, pinfo, args, train=False)
    model, testloss = train(train_loader, model, test_loader, args)
    save_model(model, args.model_name, dir_path)
    
    # model = archi(lstm_input_dim, args.lstm_hidden_dim, fc_input_dim, fc_output_dim, args.lstm_size).to(device)
    # model = load_model(model, args.model_name, dir_path)
    # single_test(model, 1, args)

    # test_loader = dataloader_prep(feat_dict, exps, pinfo, args, train=False)
    # testloss = test(model, test_loader)

    #return testloss
    # args_namespace = argparse.Namespace()
    args_dict = vars(args)
    # print(type(args_dict))
    args_dict['test_loss'] = f'{testloss:.6f}'
    args_dict.pop('dir_path')
    # print(args_dict)
    args_series = pd.Series(args_dict)
    args_df = args_series.to_frame().transpose()
    # print(args_df)

    exp_log_filepath = os.path.join(dir_path,'saved_models','experiment_log2.pkl')
    if os.path.exists(exp_log_filepath):
        exp_log = pd.read_pickle(exp_log_filepath)
        exp_log = exp_log.append(args_df).reset_index(drop=True)
        pd.to_pickle(exp_log, exp_log_filepath)
        print(exp_log)
    else:
        pd.to_pickle(args_df, exp_log_filepath)
        print(args_df)
    

    