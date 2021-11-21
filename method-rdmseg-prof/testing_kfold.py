import os
import sys
sys.path.append(os.path.abspath(''))
import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
plt.rcParams["figure.dpi"] = 300
import matplotlib
matplotlib.rcParams['font.sans-serif'] = "Arial"
matplotlib.rcParams['font.family'] = "sans-serif"
import util
from util_method import pearson_corr_loss

def single_test(model, device, songurl, feat_dict, exps): #, fold_i, args, filename_prefix=None):
    '''
        feat_dict - of the test fold.
        exps - pruned exps
    '''
    # features - audio
    testfeat = feat_dict[songurl]

    # labels
    all_exps_of_songurl_bool = exps['songurl']  == songurl
    all_exps_of_songurl = exps[all_exps_of_songurl_bool]

    testlabels = all_exps_of_songurl['labels'].to_list()
    testprofiles = all_exps_of_songurl['profile'].to_list()
    # print(len(testfeat))
    # print(len(testlabel))
    losses = {'r': [], 'mse': []}
    pred_n_gts = {}
    
    # c_fig, c_axs = plt.subplots(len(testprofiles),sharex=True)
    # a_fig, a_axs = plt.subplots(len(testprofiles),sharex=True)

    with torch.no_grad():
        # iterate through each profile category
        for j in np.arange(len(testlabels)):

            testlabel = testlabels[j]
            profile = testprofiles[j]
            if hasattr(profile, '__iter__'):
                profile = list(profile)
            else:
                profile = [profile]

            testprofiles_repeat = [profile for a in np.arange(len(testfeat))]
            # print(np.shape(testprofiles_repeat))
            testinput = np.concatenate((testfeat, testprofiles_repeat),axis=1)

            testinput = torch.from_numpy(testinput)
            # print('shape of testlabel: ', testlabel.shape)
            testlabel = torch.from_numpy(testlabel)

            testinput = testinput.unsqueeze(0)
            testlabel = testlabel.unsqueeze(0)

            feature, label = testinput.to(device).float(), testlabel.to(device).float()
            # model = load_model(model, args.model_name, args.dir_path)

            # forward pass
            output = model(feature)
            output = output.squeeze(1)
    
            # MSE Loss calculation
            loss_mse = F.mse_loss(output, label)
            loss_r = pearson_corr_loss(output, label)
            losses['mse'].append(loss_mse.item())
            losses['r'].append(loss_r.item())

            pred_n_gts[testprofiles[j]] = {}
            pred_n_gts[testprofiles[j]]['mse'] = round(loss_mse.item(),4)
            pred_n_gts[testprofiles[j]]['r'] = round(loss_r.item(),4)
            pred_n_gts[testprofiles[j]]['pred'] = output.squeeze().cpu().numpy()
            pred_n_gts[testprofiles[j]]['gtruth'] = testlabels[j]
        
    return np.mean(losses['mse']), np.mean(losses['r']), pred_n_gts

def plot_pred_n_gts(pred_n_gts, songurl, args, savepath, filename_prefix=None):
    '''
    WARNING: assumes taking into consideration only a single profile type. else just chooses the first profile type.
    '''
    profile_type = args.conditions[0]
    testprofiles = pred_n_gts.keys()
    # print(testprofiles)
    fig_h = len(testprofiles)*3
    c_fig, c_axs = plt.subplots(len(testprofiles),sharex=True, figsize=(8,fig_h))
    a_fig, a_axs = plt.subplots(len(testprofiles),sharex=True, figsize=(8,fig_h))
    
    if not hasattr(c_axs, '__iter__'):
            c_axs = [c_axs]
            a_axs = [a_axs]

    for j, profile in enumerate(testprofiles):
        
        c_axs[j].plot(pred_n_gts[profile]['pred']) #, label='prediction')
        c_axs[j].plot(pred_n_gts[profile]['gtruth']) #, label='ground truth')
        
        c_axs[j].set_title(f"mse: {pred_n_gts[profile]['mse']} || r: {pred_n_gts[profile]['r']}")
        c_axs[j].set_ylim([-1, 1])
        c_axs[j].set_ylabel(f"{util.r_mapper_dict[profile_type][profile]}")

        # temp_plot_a = plot_pred_against(output, testlabels[j])
        a_axs[j].scatter(pred_n_gts[profile]['gtruth'], pred_n_gts[profile]['pred'], marker='x')
        a_axs[j].set_ylim([-1, 1])
        a_axs[j].set_xlim([-1, 1])
        a_axs[j].set_ylabel(f"{util.r_mapper_dict[profile_type][profile]}")
    
    # dir_path = os.path.dirname(os.path.realpath(__file__))

    c_fig.suptitle(f'{songurl} || {profile_type}')
    c_fig.legend(['prediction', 'ground truth'])
    c_fig.tight_layout(h_pad=0.5)
    c_fig.savefig(os.path.join(savepath, f'predictions/{filename_prefix}{songurl}_prediction.png'))
    # plt.show()
    plt.close(c_fig)

    a_fig.suptitle(f'{songurl} || {profile_type}', y=0.91)
    c_fig.tight_layout(h_pad=0.5)
    a_fig.savefig(os.path.join(savepath, f'predictions/{filename_prefix}{songurl}_y_vs_yhat.png'))
    # plt.show()
    plt.close(a_fig)




    

if __name__ == "__main__":
    
    import sys
    sys.path.append(os.path.abspath(''))
    sys.path.append(os.path.abspath('processing'))
    print(sys.path)
    import argparse

    import pandas as pd
    from util_method import load_model

    from networks import lstm_double as archi_lstm
    from networks import Three_FC_layer as archi_linear

    from ave_exp_by_prof import ave_exps_by_profile

    ########################
    ####    Argparse    ####
    ########################

    dir_path = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default=dir_path)
    parser.add_argument('--master', type=int, default=0) # use int instead of boolean.
    parser.add_argument('--linear', type=bool, default=False)
    parser.add_argument('--affect_type', type=str, default='valences', help='Can be either "arousals" or "valences"')
    parser.add_argument('--model_name', type=str, default='hd512_mse_smooth15_age', help='Name of folder plots and model will be saved in')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--conditions', nargs='+', type=str, default=['age'])#['play_instrument', 'training', 'training_duration']
    # age, gender, master, country_enculturation, country_live, fav_music_lang, fav_genre, play_instrument, training, training_duration

    args = parser.parse_args()
    if args.master == 0:
        save_models_foldername = 'saved_models'
    else:
        save_models_foldername = 'saved_models_m'
    if args.linear:
        setattr(args, 'model_name', f'linear_{args.affect_type[0]}_p_{args.model_name}')
        exp_log_filepath = os.path.join(dir_path,'saved_models','test_log_linear.pkl')
        archi = archi_linear
    else:
        setattr(args, 'model_name', f'{args.affect_type[0]}_p_{args.model_name}')
        exp_log_filepath = os.path.join(dir_path,'saved_models','test_log_lstm.pkl')
        archi = archi_lstm
    print(args)

    savepath = os.path.join(dir_path,save_models_foldername, args.model_name)
    load_model_name = f'{args.affect_type[0]}_p_semoga_akhir'
    loadpath = os.path.join('/home/meowyan/Documents/emotion/method-rdmseg/', save_models_foldername, load_model_name)
    print('load path: ', loadpath)



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

    #########################
    ####    Load Data    ####
    #########################

    pinfo = util.load_pickle('data/pinfo_numero.pkl')
    original_exps = pd.read_pickle('data/exps_ready3.pkl')
    '''
    pick out the master participants only and test.
    '''
    # pinfo_master = pinfo[pinfo['master'] == 1.0]
    exps = ave_exps_by_profile(original_exps, pinfo, args.affect_type, args.conditions)
    print('shape of exps: ', exps.shape)
    print('shape of pinfo: ', pinfo.shape)

    '''
    5 FOLD CROSS VALIDATION LEGGO
    '''
    loss_log_folds = {'mse':[], 'r':[]}
    num_folds = 1
    for fold_i in range(num_folds):
        
        ## Load test features for fold_i
        test_feat_dict = util.load_pickle(f'data/folds/test_feats_{fold_i}.pkl')
        ## MODEL
        input_dim = list(test_feat_dict.values())[0].shape[1] + len(args.conditions) #724 # 1582 
        model = archi(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)
        model.float()
        # print(model)

        load_model(model, loadpath, f'{load_model_name}_{fold_i}')
        
        losses = {'mse':[], 'r':[]}
        for songurl in test_feat_dict.keys():
            
            mse, r, pred_n_gts = single_test(model, device, songurl, test_feat_dict, exps)
            losses['mse'].append(mse)
            losses['r'].append(r)
            plot_pred_n_gts(pred_n_gts, songurl, args)
            break
        
        print(f"mse: {np.mean(losses['mse']):.4f} || r: {np.mean(losses['r']):.4f}")
        loss_log_folds['mse'].append(np.mean(losses['mse']))
        loss_log_folds['r'].append(np.mean(losses['r']))


    ave_loss = {
        'mse': round(np.mean(loss_log_folds['mse']),4),
        'r': round(np.mean(loss_log_folds['r']),4)
    }


    print(f"mean of folds || mse: {ave_loss['mse']} || r: {ave_loss['r']}")

    ss = pd.Series(ave_loss)
    df = ss.to_frame().transpose()
    print(df)
    # if os.path.exists(exp_log_filepath):
    #     exp_log = pd.read_pickle(exp_log_filepath)
    #     exp_log = exp_log.append(df).reset_index(drop=True)
    #     pd.to_pickle(exp_log, exp_log_filepath)
    #     print(exp_log)
    # else:
    #     pd.to_pickle(df, exp_log_filepath)
    #     print(df)