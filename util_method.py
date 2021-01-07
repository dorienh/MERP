import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

def save_model(model, model_name, dir_path):

    path_to_save = os.path.join(dir_path, 'saved_models', f"{model_name}", f"{model_name}.pth")
    torch.save(model.state_dict(), path_to_save)
    # loss_fig.savefig(os.path.join(args.model_path, f"{model_name}_loss_plot.png"))

def load_model(model, model_name, dir_path):

    path_to_load = os.path.join(dir_path, 'saved_models', f"{model_name}", f"{model_name}.pth")
    # print(path_to_load)
    model.load_state_dict(torch.load(path_to_load))
    model.eval() # assuming loading for eval and not further training. (does not save optimizer so shouldn't continue training.)
    return model

def plot_pred_comparison(output, label, mseloss, rloss=None):
    plt.plot(output.cpu().numpy(), label='prediction')
    plt.plot(label.cpu().numpy(), label='ground truth')
    plt.legend()
    if not rloss:
        plt.title(f'Prediction vs Ground Truth || mse: {mseloss}')
    else:
        plt.title(f'Prediction vs Ground Truth || mse: {mseloss:.5} || r: {rloss:.5}')
    return plt

def plot_pred_against(output, label):
    actual = label.cpu().numpy()
    predicted = output.squeeze().cpu().numpy()
    # print(np.shape(actual))
    # print(np.shape(predicted))
    plt.scatter(actual, predicted)
    return plt

# standardize the features
def standardize(feat_dict): # all together
    mean_sum = np.zeros(1582)
    std_sum = np.zeros(1582)
    for songurl, audio_feat in feat_dict.items():
        # print(songurl)
        # print(np.shape(audio_feat))
        mean_sum += np.mean(audio_feat, axis=0)
        # print(np.shape(mean))
        std_sum += np.std(audio_feat,axis=0)
    
    mean = mean_sum/len(feat_dict)
    std = std_sum/len(feat_dict)

    for songurl, audio_feat in feat_dict.items():
        standard_feat = (audio_feat - mean)/std
        # print(standard_feat)
        feat_dict[songurl] = standard_feat

    return feat_dict


def combine_similar_pinfo(pinfo, exps, args):
    '''
    averages or finds the median of labels should they be 
    '''
    print(args.mean, args.median)
    # print(pinfo.head)
    desired_columns = ['workerid', *args.conditions]
    selected_pinfo = pinfo[desired_columns]
    # print(selected_pinfo.head())

    # a list to keep every unique trial and pinfo type
    unique_trials = []
    
    for values, group in selected_pinfo.groupby(args.conditions):
        # print('grouby')
        # print(group['workerid'])
        group_exps = exps[exps['workerid'].isin(group['workerid'])]
        # print(group_exps)

        for song, song_group in group_exps.groupby('songurl'):
            # print(song)
            # print(len(song_group))
            if len(song_group) <= 1:
                # only one trial exists, so we need not do anything to it
                unique_trials.append(song_group.iloc[0])
                # print(type(song_group.iloc[0]))
            else:
                # average/median the labels
                label_list = song_group[args.affect_type].to_numpy()
                
                # print(type(label_list[0]))
                
                if args.mean:
                    label_agg = np.mean(label_list)
                if args.median:
                    # change to 
                    label_list = [list(label) for label in label_list]
                    label_agg = np.median(label_list, axis=0)
                    # print(np.shape(label_agg))
                
                trial = song_group.iloc[0]
                trial.at[args.affect_type] = label_agg
                # print(trial)
                unique_trials.append(trial)
    # list of series to dataframe
    print('length of unique_trials: ', len(unique_trials))
    return pd.DataFrame(unique_trials)
    
        
def combine_no_profile(exps, args):

    print(args.mean, args.median)

    combined_list = []

    for songurl, group in exps.groupby('songurl'):
        label_list = group[args.affect_type].to_numpy()
        if args.mean:
            label_agg = np.mean(label_list)
        if args.median:
            # change to 
            label_list = [list(label) for label in label_list]
            label_agg = np.median(label_list, axis=0)

        trial = group.iloc[0]
        trial.at[args.affect_type] = label_agg
        combined_list.append(trial)
    
    return pd.DataFrame(combined_list)


