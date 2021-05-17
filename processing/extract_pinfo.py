
import pandas as pd
import numpy as np
from processing_util import batch_names, filepath_dict, csv2df
import os
import sys
# sys.path.append(os.path.abspath('../..'))
sys.path.append(os.path.abspath(''))
# print(sys.path)
import util


def participant_info_df_creation(df, batchnum):
    """
    extracts personal information from DataFrame of amazon turk data and outputs a DataFrame, one row for one participant.
    """
    if batchnum is '7' or batchnum is '8': # if not master
        master = 0.0
    else: # if master
        master = 1.0
    template = {
        'workerid': df['WorkerId'],
        'batch': batchnum,
        'master': master,
        'age': df['Answer.age'],
        'gender': df['Answer.gender'],
        'residence': df['Answer.country-live'],
        'enculturation': df['Answer.country-enculturation'],
        'language': df['Answer.fav-music-language'],
        'genre': df['Answer.genre'],
        'instrument': df['Answer.play-instrument'],
        'training': df['Answer.training'],
        'duration': df['Answer.training-duration'].replace('{}', 0)
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
'''
if __name__ == '__main__':
    pinfo = collate_all_batches_participant_info()
    # print(pinfo)

    exps = util.load_pickle('data/exps_ready3.pkl')

    # https://datascience.stackexchange.com/questions/47562/multiple-filtering-pandas-columns-based-on-values-in-another-column
    def pair_columns(df, col1, col2):
        return df[col1] + df[col2]

    def paired_mask(df1, df2, col1, col2):
        return pair_columns(df1, col1, col2).isin(pair_columns(df2, col1, col2))

    identical = pinfo.loc[paired_mask(pinfo, exps, "workerid", "batch")]
    
    identical.to_pickle('data/mediumrare/semipruned_pinfo.pkl')
'''

if __name__ == '__main__':
    # main()
    pinfo = collate_all_batches_participant_info()


    print(pinfo['master'].value_counts())
    print('unique wid count: ', len(pinfo['workerid'].unique()))

    # removing rejected workerids

    duplicate_wids = pinfo.duplicated(subset=['workerid'],keep=False)
    
    duplicate_pinfo = pinfo[duplicate_wids].sort_values('workerid')
    print(duplicate_pinfo)
    print('num duplicates: ', len(duplicate_pinfo))
    print('master count: \n', duplicate_pinfo.master.value_counts())
    # print(duplicate_pinfo.master.value_counts())
    not_conflicting_bool = duplicate_pinfo.duplicated(subset=['workerid','age', 'gender', 'residence', 'enculturation', 'language', 'genre', 'instrument', 'training', 'duration'])
    not_dups = duplicate_pinfo[not_conflicting_bool].sort_values('workerid')

    # print('meow', not_dups)

    # union of the series 
    union = pd.Series(np.union1d(duplicate_pinfo.workerid, not_dups.workerid)) 
    
    # intersection of the series 
    intersect = pd.Series(np.intersect1d(duplicate_pinfo.workerid, not_dups.workerid)) 
    
    # uncommon elements in both the series  
    notcommonseries = union[~union.isin(intersect)] 
    print('num disqualified dups: ', len(notcommonseries))
    # manually adding the 6 incomplete participants
    todelete = notcommonseries.append(pd.Series(['A3KPQ7L5FS8SD6','A21SF3IKIZB0VN','A2Z3I9XW0SHBPY','A3DKB1786IV19A','A3681W483PXK3P','A2WWYVKGZZXBOB']))

    temp = pinfo.loc[pinfo.workerid.str.contains('|'.join(['A3KPQ7L5FS8SD6','A21SF3IKIZB0VN','A2Z3I9XW0SHBPY','A3DKB1786IV19A','A3681W483PXK3P','A2WWYVKGZZXBOB']))]
    print(temp)
    unique_pinfo = pinfo[~pinfo['workerid'].isin(todelete)]
    # print('unique_pinfo.shape: ', unique_pinfo.shape)

    print('unique_pinfo master count')
    print(unique_pinfo.master.value_counts())
    # print(duplicate_pinfo.master.value_counts())

    # identify erroneous profiles

    # fishy training durations...
    pinfo_td = unique_pinfo['duration'].astype(int)
    err_p1 = unique_pinfo[(pinfo_td < 0) | (pinfo_td > 100)]
    print('num wids with weird training duration: ', len(err_p1))

    err_p2 = unique_pinfo.loc[(unique_pinfo['duration'].astype(int)>0) & (unique_pinfo['training']=='No')]
    print('num wids with no training but with training duration: ', len(err_p2))
    todelete = err_p1.append(err_p2)

    # print(todelete)
    
    clean_pinfo = unique_pinfo[~unique_pinfo['workerid'].isin(todelete.workerid)]
    # print(clean_pinfo.shape)

    # print(clean_pinfo.gender.value_counts())
    # clean_pinfo = clean_pinfo.drop_duplicates(subset=['workerid'])
    
    temp = clean_pinfo.duplicated(subset=['workerid','age', 'gender', 'residence', 'enculturation', 'language', 'genre', 'instrument', 'training', 'duration'],keep=False)
    print(clean_pinfo[temp])
    # print(clean_pinfo.iloc[100])
    # print(clean_pinfo[clean_pinfo.workerid.duplicated(keep=False)])


    clean_pinfo.to_pickle('data/mediumrare/semipruned_pinfo.pkl')


