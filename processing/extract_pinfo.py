
import pandas as pd
import numpy as np
from processing_util import batch_names, filepath_dict, csv2df

def participant_info_df_creation(df, batchnum):
    """
    extracts personal information from DataFrame of amazon turk data and outputs a DataFrame, one row for one participant.
    """
    if batchnum is '7' or batchnum is '8':
        master = 0
    else:
        master = 1
    template = {
        'workerid': df['WorkerId'],
        'batch': batchnum,
        'master': master,
        'age': df['Answer.age'],
        'country_enculturation': df['Answer.country-enculturation'],
        'country_live': df['Answer.country-live'],
        'fav_music_lang': df['Answer.fav-music-language'],
        'gender': df['Answer.gender'],
        'fav_genre': df['Answer.genre'],
        'play_instrument': df['Answer.play-instrument'],
        'training': df['Answer.training'],
        'training_duration': df['Answer.training-duration'].replace('{}', 0)
    }
    return pd.DataFrame(template)

def collate_all_batches_participant_info():
    participant_info_df_list = []
    for batchname in filepath_dict.keys():
        # print(csv_path)
        df = csv2df(filepath_dict[batchname])
        '''
        TO DO: select only those that belong in the namelist!
        '''
        participant_info_df_list.append(participant_info_df_creation(df, batch_names[batchname]))

    return pd.concat(participant_info_df_list, ignore_index=True)


def main():
    participant_info = collate_all_batches_participant_info()
    print(participant_info.head())
    participant_info.to_pickle('data/mediumrare/unpruned_pinfo.pkl')


if __name__ == '__main__':
    # main()
    pinfo = collate_all_batches_participant_info()

    # removing rejected workerids

    duplicate_wids = pinfo.duplicated(subset=['workerid'],keep=False)
    duplicate_pinfo = pinfo[duplicate_wids]
    not_conflicting_bool = duplicate_pinfo.duplicated(subset=['age', 'country_enculturation', 'country_live', 'fav_music_lang', 'gender', 'fav_genre', 'play_instrument', 'training', 'training_duration'])
    not_dups = duplicate_pinfo[not_conflicting_bool]

    # print(not_dups)

    # union of the series 
    union = pd.Series(np.union1d(duplicate_pinfo.workerid, not_dups.workerid)) 
    
    # intersection of the series 
    intersect = pd.Series(np.intersect1d(duplicate_pinfo.workerid, not_dups.workerid)) 
    
    # uncommon elements in both the series  
    notcommonseries = union[~union.isin(intersect)] 
    # manually adding the 6 incomplete participants
    todelete = notcommonseries.append(pd.Series(['A3KPQ7L5FS8SD6','A21SF3IKIZB0VN','A2Z3I9XW0SHBPY','A3DKB1786IV19A','A3681W483PXK3P','A2WWYVKGZZXBOB']))

    unique_pinfo = pinfo[~pinfo['workerid'].isin(todelete)]
    print('unique_pinfo.shape: ', unique_pinfo.shape)

    print(unique_pinfo.master.value_counts())
    print(duplicate_pinfo.master.value_counts())

    # identify erroneous profiles

    # fishy training durations...
    pinfo_td = unique_pinfo['training_duration'].astype(int)
    err_p1 = unique_pinfo[(pinfo_td < 0) | (pinfo_td > 100)]
    print('num wids with weird training duration: ', len(err_p1))

    err_p2 = unique_pinfo.loc[(unique_pinfo['training_duration'].astype(int)>0) & (unique_pinfo['training']=='No')]
    print('num wids with no training but with training duration: ', len(err_p2))
    todelete = err_p1.append(err_p2)
    
    print(todelete)
    
    clean_pinfo = unique_pinfo[~unique_pinfo['workerid'].isin(todelete.workerid)]
    print(clean_pinfo.shape)

    print(clean_pinfo.gender.value_counts())

    clean_pinfo.to_pickle('data/mediumrare/semipruned_pinfo.pkl')


