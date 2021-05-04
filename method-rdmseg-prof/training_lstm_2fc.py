'''
since we are randomly selected segments of each song in each epoch, 
the number of epochs is explosive so by a high probability, every entire song is accounted for. lol.
'''

import os
import sys
# sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath('processing'))
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
from util_method import save_model, load_model, plot_pred_against, plot_pred_comparison, standardize, windowing, reverse_windowing
### to edit accordingly.
from dataset import rdm_dataset as dataset_class
### to edit accordingly.
from networks import lstm_single_2fc as archi

from ave_exp_by_prof import ave_exps_by_profile

#####################
####    Train    ####
#####################
def train(train_loader, model, valid_loader, args):
    
    
    loss_log = {
        'train_mse' : [],
        'train_r' : [],
        'valid_mse' : [],
        'valid_r' : []
    }
    '''
        intial round
    '''
    with torch.no_grad():
        model.eval()
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
    n_epochs_least = 100
    n_epochs_stop = 50
    epochs_no_improve = 0
    min_val_loss = np.Inf
    early_stop = False

    for epoch in np.arange(args.num_epochs):
        # valid_loss = 0 # for early stopping
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
            # if torch.isnan(loss_r):
            #     loss = loss_mse
            # else:
            loss = loss_mse*args.mse_weight + loss_r

            # backward pass
            loss.backward(retain_graph=True)
            # update parameters
            optimizer.step()

            # record training loss
            epoch_loss_log['mse'].append(loss_mse.item())
            epoch_loss_log['r'].append(loss_r.item())

            print(f'Epoch: {epoch} || Batch: {batchidx}/{numbatches} || mse = {loss_mse.item():5f} || r = {loss_r.item():5f} || loss = {loss.item():5f}', end = '\r')


            # val_loss += loss.item()
        
            
        # log average loss
        aveloss_mse = np.average(epoch_loss_log['mse'])
        aveloss_r = np.average(epoch_loss_log['r'])
        # print(f'Initial round without training || mse = {aveloss_mse:.2f} || r = {aveloss_r:.2f}')
        loss_log['train_mse'].append(aveloss_mse)
        loss_log['train_r'].append(aveloss_r)
        print(' '*200)

        epoch_duration = time.time() - start_time
        print(f'Epoch: {epoch:3} || mse: {aveloss_mse:8.5f} || r: {aveloss_r:8.5f} || time taken (s): {epoch_duration:8f}')
        
        valid_ave_mse, valid_ave_r, valid_loss  = test(model, valid_loader)

        loss_log['valid_mse'].append(valid_ave_mse)
        loss_log['valid_r'].append(valid_ave_r)


        # val_loss = val_loss / len(train_loader)


        if valid_loss < min_val_loss:
            # print('meow')
            epochs_no_improve = 0
            min_val_loss = valid_loss
            best_epoch = epoch
            best_model = model
            best_valid_ave_mse = valid_ave_mse
            best_valid_ave_r = valid_ave_r
            print(f'BEST!! epoch: {epoch:3}, train_mse: {aveloss_mse:8.5f}, valid_mse: {valid_ave_mse:8.5f}, train_r: {aveloss_r:8.5f}, valid_r: {valid_ave_r:8.5f}')

        else:
            epochs_no_improve += 1
        # print(epochs_no_improve)
        # iter += 1
        if epoch > n_epochs_least and epochs_no_improve == n_epochs_stop:
            print('Early stopping!' )
            early_stop = True
        
        if early_stop:
            print("Stopped")
            break

    # test_ave_mse, test_ave_r, _  = test(model, test_loader)


    # plot loss against epochs
    plt.plot(loss_log['train_mse'][1::], label='training loss mse')
    plt.plot(loss_log['valid_mse'], label='validation loss mse')
    plt.plot(loss_log['train_r'][1::], label='training loss r')
    plt.plot(loss_log['valid_r'], label='validation loss r')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.ylim([-1, 1])
    plt.legend()
    plt.title(f"Loss || init mse:{loss_log['train_mse'][0]:.3f} | r:{loss_log['train_r'][0]:.3f} || valid mse: {best_valid_ave_mse:.3f} | r: {best_valid_ave_r:.3f}")
    plt.savefig(os.path.join(args.dir_path, 'saved_models', f'{args.model_name}', 'loss_plot.png'))
    plt.close()

    return best_model, best_valid_ave_mse, best_valid_ave_r, best_epoch
    # return model, test_ave_mse, test_ave_r

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
            # print(feature)
            # forward pass
            output = model(feature)
            # MSE Loss calculation
            # print(output.squeeze())
            # print(label.squeeze())
            loss_mse = nn.MSELoss()(output.squeeze(), label.squeeze())
            loss_r = pearson_corr_loss(output.squeeze(), label.squeeze())
            # print('pearson input: ', output.squeeze(), '\n', label.squeeze())
            # loss = loss_mse + loss_r
            # print(loss)
            epoch_loss_log['mse'].append(loss_mse.item())
            epoch_loss_log['r'].append(loss_r.item())
            # print('where is the nan coming from?')
            # print(loss_mse.item())
            # print(loss_r.item())
            # break
    
    test_ave_mse = np.average(epoch_loss_log['mse'])
    test_ave_r = np.average(epoch_loss_log['r'])
    sum_test = test_ave_mse + test_ave_r
    print(f'average test lost (per batch): mse: {test_ave_mse:4f} r: {test_ave_r:4f} sum: {sum_test:4f}')
    return test_ave_mse, test_ave_r, sum_test


def single_test(model, songurl, feat_dict, exps, args, filename_prefix=None):
    '''
        exps - the original exps with many workers
    '''
    # print(songurl)
    # features - audio
    testfeat = feat_dict[songurl]
    # features - exps
    # labels
    all_exps_of_songurl_bool = exps['songurl']  == songurl
    all_exps_of_songurl = exps[all_exps_of_songurl_bool]
    # testlabel = exps.at[songurl,'labels']
    # print(all_exps_of_songurl)
    testlabels = all_exps_of_songurl['labels'].to_list()
    testprofiles = all_exps_of_songurl['profile'].to_list()
    # print(len(testfeat))
    # print(len(testlabel))

    testinput_w = windowing(testfeat, args.lstm_size, step_size=args.lstm_size)
    
    c_fig, c_axs = plt.subplots(len(testprofiles),sharex=True)
    a_fig, a_axs = plt.subplots(len(testprofiles),sharex=True)


    for j in np.arange(len(testlabels)):

        testlabel_w = windowing(testlabels[j], args.lstm_size, step_size=args.lstm_size)

        outputs = []
        loss_mse_list = []
        loss_r_list = []

        with torch.no_grad():
            for i in np.arange(len(testinput_w)):
                
                profile = testprofiles[j]
                if hasattr(profile, '__iter__'):
                    profile = list(profile)
                else:
                    profile = [profile]

                testprofiles_repeat = [profile for a in np.arange(len(testinput_w[i]))]
                # print(np.shape(testprofiles_repeat))
                # print(np.shape(testinput_w[i]))
                testinput_i = np.concatenate((testinput_w[i], testprofiles_repeat),axis=1)
                
                # print(np.shape(testinput_i))
                testlabel_i = testlabel_w[i]

                testinput_i = torch.from_numpy(testinput_i)
                testlabel_i = torch.from_numpy(testlabel_i)

                testinput_i = testinput_i.unsqueeze(0)
                testlabel_i = testlabel_i.unsqueeze(0)

                feature, label = testinput_i.to(device).float(), testlabel_i.to(device).float()
                # model = load_model(model, args.model_name, args.dir_path)

                # forward pass
                output = model(feature)
                
                # MSE Loss calculation
                # loss = criterion(output.squeeze(), label.squeeze())
                loss_mse = nn.MSELoss()(output.squeeze(), label.squeeze())
                loss_mse_list.append(loss_mse.item())
                loss_r = pearson_corr_loss(output.squeeze(), label.squeeze())
                
                loss_r_list.append(loss_r.item())
                # loss = loss_mse + loss_r

                output = output.squeeze()
                output = output.cpu().numpy()
                output = np.atleast_1d(output)
                # https://stackoverflow.com/questions/35617073/python-numpy-how-to-best-deal-with-possible-0d-arrays
                outputs.append(output)

        # print(loss.item())
        output = reverse_windowing(outputs, args.lstm_size, step_size=args.lstm_size)

        dir_path = os.path.dirname(os.path.realpath(__file__))

        # temp_plot_c = plot_pred_comparison(output, testlabels[j], np.mean(loss_mse_list), np.mean(loss_r_list))
        c_axs[j].plot(output) #, label='prediction')
        c_axs[j].plot(testlabels[j]) #, label='ground truth')
        rloss = np.mean(loss_r_list)
        mseloss = np.mean(loss_mse_list)
        c_axs[j].set_title(f'mse: {mseloss:.5} || r: {rloss:.5}')
        c_axs[j].set_ylim([-1, 1])

        
        # temp_plot_a = plot_pred_against(output, testlabels[j])
        a_axs[j].scatter(testlabels[j], output, marker='x')
        

    c_fig.suptitle(f'{songurl}')
    c_fig.legend(['prediction', 'ground truth'])
    c_fig.savefig(os.path.join(dir_path, 'saved_models', f'{args.model_name}/predictions/{filename_prefix}{songurl}_prediction.png'))
    plt.close(c_fig)

    
    a_fig.suptitle(f'{songurl}')
    a_fig.savefig(os.path.join(dir_path, 'saved_models', f'{args.model_name}/predictions/{filename_prefix}{songurl}_y_vs_yhat.png'))
    plt.close(a_fig)



if __name__ == "__main__":
    
    ########################
    ####    Argparse    ####
    ########################

    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=dir_path)
    parser.add_argument('--affect_type', type=str, default='valences', help='Can be either "arousals" or "valences"')
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--model_name', type=str, default='test123', help='Name of folder plots and model will be saved in')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--lstm_size', type=int, default=10)
    # parser.add_argument('--step_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--mse_weight', type=float, default=1.0)

    parser.add_argument('--conditions', nargs='+', type=str, default=['age'])#['play_instrument', 'training', 'training_duration']
    # age, gender, master, country_enculturation, country_live, fav_music_lang, fav_genre, play_instrument, training, training_duration

    args = parser.parse_args()
    setattr(args, 'model_name', f'{args.affect_type[0]}_p_{args.model_name}')
    print(args)

    # check if folder with same model_name exists. if not, create folder.
    os.makedirs(os.path.join(dir_path,'saved_models', args.model_name), exist_ok=True)
    os.makedirs(os.path.join(dir_path,'saved_models', args.model_name, 'predictions'), exist_ok=True)

    #########################
    ####    Load Data    ####
    #########################

    # load the data 
    # read audio features from pickle
    train_feat_dict = util.load_pickle('data/train_feats.pkl')
    valid_feat_dict = util.load_pickle('data/valid_feats.pkl')
    test_feat_dict = util.load_pickle('data/test_feats.pkl')
    # read labels from pickle
    
    # exps = pd.read_pickle('data/exps_std_a_profile_ave.pkl')
    pinfo = util.load_pickle('data/pinfo_numero.pkl')
    original_exps = pd.read_pickle('data/exps_ready3.pkl')
    exps = ave_exps_by_profile(original_exps, pinfo, args.affect_type, args.conditions)
    # print(exps.head())
    
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
        
        if torch.isnan(cost):
            return torch.tensor([0]).to(device)
        else:
            return cost*-1

    ###########################
    ####    Dataloading    ####
    ###########################

    def dataloader_prep(feat_dict, exps, args):
        params = {'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': args.num_workers}
        # prepare data for testing
        
        dataset = dataset_class(feat_dict, exps, seq_len=args.lstm_size)
        loader = DataLoader(dataset, **params)
        return loader


    ###########################
    ####    Model param    ####
    ###########################

    ## MODEL
    input_dim = list(train_feat_dict.values())[0].shape[1] + len(args.conditions) #724 # 1582 
    # model = archi(input_dim=input_dim, hidden_dim=args.hidden_dim, kernel_size=3).to(device)
    model = archi(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
    model.float()
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    ########################
    ####    Training    ####
    ########################
    

    train_loader = dataloader_prep(train_feat_dict, exps, args)
    valid_loader = dataloader_prep(valid_feat_dict, exps, args)
    test_loader = dataloader_prep(test_feat_dict, exps, args)
    
    model, val_ave_mse, val_ave_r, num_epochs = train(train_loader, model, valid_loader, args)

    save_model(model, args.model_name, dir_path)

    #######################
    ####    Testing    ####
    #######################

    # model = archi(input_dim).to(device)
    # model = load_model(model, args.model_name, dir_path)
    test_ave_mse, test_ave_r, sum_test  = test(model, test_loader)

    for songurl in util.testlist:
        single_test(model, songurl, test_feat_dict, exps, args)

    for songurl in util.trainlist[0:5]:
        single_test(model, songurl, train_feat_dict, exps, args, 'train')
        
    # single_test(model, '0505_58', exps, args)

    # logging

    args_dict = vars(args)
    # print(type(args_dict))
    args_dict['num_epochs'] = num_epochs
    args_dict['v_mse'] = f'{val_ave_mse:.6f}'
    args_dict['v_r'] = f'{val_ave_r:.6f}'
    args_dict['v_loss'] = f'{val_ave_mse+val_ave_r:.6f}'

    args_dict['t_mse'] = f'{test_ave_mse:.6f}'
    args_dict['t_r'] = f'{test_ave_r:.6f}'
    args_dict['t_loss'] = f'{sum_test:.6f}'
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