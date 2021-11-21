
import os
import glob
import pandas as pd
import numpy as np
import sys
sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath('../'))
sys.path.append(os.path.abspath(''))
import util

import matplotlib.pyplot as plt


def average_exps_by_songurl(exps, affect_type, profile=None, profile_cat=None):
    ave_labels = {}
    for songurl, group in exps.groupby('songurl'):
        # print(f'num wids for {songurl}: {len(group)}')
        ave = group[affect_type].mean()
        ave_labels[songurl] = ave
        if profile: # plot
            plot_ave_label_cat_songurl(ave, songurl, profile, profile_cat, len(group))
    return ave_labels

def plot_ave_label_cat_songurl(ave_label, songurl, profile, profile_cat, num_wids):

    plt.plot(ave_label)
    plt.title(f'profile: {profile} || cat: {profile_cat} || song: {songurl} || num wids: {num_wids}')
    plt.savefig(f'analysis/plots/profile_cat_ave_labels/{profile}_{profile_cat}_{songurl}.png')
    plt.close()

def ave_exps_by_profile(exps, pinfo, affect_type, profile):

    ofinterest = pinfo[profile+['workerid']]

    combined_songurl_list = []
    combined_labels_list = []
    combined_prof_list = []

    for profile_cat, group_p in ofinterest.groupby(profile):
        # print('profile_cat: ', profile_cat)
        wids_in_profile_cat = group_p['workerid'].to_numpy()

        cat_exps_bool = exps[['workerid']].isin(wids_in_profile_cat)
        cat_exps = exps.loc[cat_exps_bool.to_numpy()]
        # print('num wids in cat: ', len(cat_exps))
        ave_cat_labels = average_exps_by_songurl(cat_exps, affect_type)
        '''
        next three lines for plotting to check for plateaus after averaging.
        '''
        # profile_cat_str = str(profile_cat).replace('.', '_')
        # profile_str = "_".join(profile)
        # ave_cat_labels = average_exps_by_songurl(cat_exps, affect_type, profile_str , profile_cat_str)
        # print('len ave_cat_labels: ', len(ave_cat_labels))

        # duplicate profile feature once for each song (54)
        profile_col = [profile_cat for _ in np.arange(len(ave_cat_labels))] 
        # profile_col = np.reshape(profile_col, (-1,len(profile)))
        # print('nya', np.shape(profile_col))

        # aved_cat_exps = list(zip(ave_cat_labels.keys(),ave_cat_labels.values(),profile_col))
        combined_songurl_list = combined_songurl_list + list(ave_cat_labels.keys())
        combined_labels_list = combined_labels_list + list(ave_cat_labels.values())
        combined_prof_list = combined_prof_list + profile_col
        
    # unzip profile information if more than one profile selected
    # if len(profile) > 1:
    #     combined_prof_list = list(zip(*combined_prof_list))

    # standardize labels 
    # alllabels = np.concatenate(combined_labels_list)
    # mean = alllabels.mean()
    # std = alllabels.std()

    # std_alllabels = (alllabels-mean) /std
    # mini = alllabels.min()
    # maxi = alllabels.max()
    # print(mini, maxi)

    # std_labels = (combined_labels_list - mean) / std
    # norm_labels = [util.normalize_01(a,mini,maxi) for a in combined_labels_list]

    df = pd.DataFrame()
    df['songurl'] = combined_songurl_list
    df['labels'] = combined_labels_list
    # for i in range(len(profile)):
    #     df[f'{profile[i]}'] = combined_prof_list[i]
    if len(np.shape(combined_prof_list)) < 2:
        df['profile'] = np.reshape(combined_prof_list, (-1,len(profile))) # combined_prof_list
    else:
        df['profile'] = combined_prof_list
    # print(df.head())

    return df

if __name__ == "__main__":

    affect_type = "arousals"
    # affect_type = "valences"

    profile = ['age', 'gender']

    exps = pd.read_pickle(os.path.join('data', 'exps_ready3.pkl'))

    pinfo = util.load_pickle('data/pinfo_numero.pkl')
    
    df = ave_exps_by_profile(exps, pinfo, affect_type, profile)
    print(df.head())
    
    import pickle

    with open(f'data/exps_std_{affect_type[0]}_profile_ave_age.pkl', 'wb') as handle:
        pickle.dump(df, handle, protocol=pickle.HIGHEST_PROTOCOL)

