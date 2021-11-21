import os
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
from util_method import pearson_corr_loss, plot_pred_against, plot_pred_comparison

def single_test(model, device, songurl, feat_dict, exps):
    '''
        exps - the original exps with many workers
    '''
    model.eval()
    # print(songurl)
    # features - audio
    testfeat = feat_dict[songurl]
    # features - exps
    # labels
    testlabel = exps.at[songurl,'labels']
    # print(len(testfeat))
    # print(len(testlabel))
    losses = {'r': [], 'mse': []}
    pred_n_gts = {}

    with torch.no_grad():
        
        testinput = torch.from_numpy(testfeat)
        testlabel = torch.from_numpy(testlabel)
        pred_n_gts['gtruth'] = testlabel

        testinput = testinput.unsqueeze(0)
        testlabel = testlabel.unsqueeze(0)

        feature, label = testinput.to(device).float(), testlabel.to(device).float()
        # model = load_model(model, args.model_name, args.dir_path)

        # forward pass
        output = model(feature)
        output = output.squeeze(1)
        
        # MSE Loss calculation
        # loss = criterion(output.squeeze(), label.squeeze())
        loss_mse = F.mse_loss(output, label)
        # loss_mse_list.append(loss_mse.item())
        loss_r = pearson_corr_loss(output, label)
        # loss_r_list.append(loss_r.item())
        # loss = loss_mse + loss_r
        losses['mse'].append(loss_mse.item())
        losses['r'].append(loss_r.item())

        pred_n_gts['mse'] = round(loss_mse.item(),4)
        pred_n_gts['r'] = round(loss_r.item(),4)
        pred_n_gts['pred'] = output.squeeze().cpu().numpy()
        

        return np.mean(losses['mse']), np.mean(losses['r']), pred_n_gts


def plot_pred_n_gts(pred_n_gts, songurl, args, filename_prefix=None):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    plt = plot_pred_comparison(pred_n_gts['pred'], pred_n_gts['gtruth'], pred_n_gts['mse'], pred_n_gts['r'])
    plt.suptitle(f'{songurl}')
    plt.savefig(os.path.join(dir_path, 'saved_models', f'{args.model_name}/predictions/{filename_prefix}{songurl}_prediction.png'))
    plt.close()

    plt = plot_pred_against(pred_n_gts['pred'], pred_n_gts['gtruth'])
    plt.suptitle(f'{songurl}')
    plt.savefig(os.path.join(dir_path, 'saved_models', f'{args.model_name}/predictions/{filename_prefix}{songurl}_y_vs_yhat.png'))
    plt.close()
