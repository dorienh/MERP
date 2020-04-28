
import pandas as pd
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
    participant_info.to_csv('data/mediumrare/unpruned_pinfo.csv')


if __name__ == '__main__':
    main()
