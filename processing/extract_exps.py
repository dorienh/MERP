import pandas as pd
import numpy as np
import re

import sys
import os
# sys.path.append(os.path.abspath(''))

from processing_util import batch_names, filepath_dict, csv2df

## start for loop here. 
def process_single_experiment(single_row, deam_dict, song_dict, batchnum):
    """
    returns a list of dictionaries, one dictionary for each song, valence and arousal triplet collected from the same participant.
    """
    songlist = np.array(single_row['Answer.random_indices'].split(','), dtype=np.int32)
    deamlist = np.array(single_row['Answer.random_deam_indices'].split(','), dtype=np.int32)

    allsongs = []
    songidx = 0
    for i in range(24):
        try:
            ars = np.array(single_row['Answer.arousals_%s'%(i+1)].split(','),dtype=np.float32)
            vl = np.array(single_row['Answer.valences_%s'%(i+1)].split(','),dtype=np.float32)

        except ValueError:
            # print('breaking out at : ', i+1)
            if i < 9:
                print('Abandon experiment: participant {} stopped at the {}th song'.format(single_row['WorkerId'],i))
                return []
            break

        ## assert that the a and v pair are of the same length.
        assert len(ars) == len(vl), 'Error: arousal and valence pair are of different length!'

        if i in [0,3,6,9]:
            # get song from deam
            # print("index %s should be deam"%(i//3))
            songurl = deam_dict[deamlist[i//3]]
        else:
            # get song from others
            # print("index %s should be others"%(songidx))
            songurl = song_dict[songlist[(songidx)]]
            songidx += 1
        onesong = {
            'workerid': single_row['WorkerId'],
            'batch': batchnum,
            'songurl': songurl.lower(),
            'arousals': ars,
            'valences': vl
            }
        allsongs.append(onesong)
    
    return allsongs

# experiment1 = process_single_experiment(single_row)
# exp1 = pd.DataFrame(experiment1)

def process_batch(df,batchnum):
    
    num_exps = df.shape[0]

    deam_dict = [re.sub('/', '_', re.findall(r'\w+/\d+\.mp3',path)[-1])[:-4] for path in df['Input.deam_music_all'][0].split(';')]
    song_dict = [re.sub('/', '_', re.findall(r'\w+/\d+\.mp3',path)[-1])[:-4] for path in df['Input.music_all'][0].split(';')]

    experiment_list = []

    for i in range(num_exps):
        single_row = df.iloc[i]
        
        single_experiment = process_single_experiment(single_row, deam_dict, song_dict, batchnum)
        
        experiment_list += single_experiment
    
    return pd.DataFrame(data=experiment_list)

# exp_list = process_batch(four)
# df = pd.DataFrame(data=exp_list)
# df.to_csv(r'/Users/koen/Desktop/Amazon/batch4exps.csv', index=False, header=True)

def collate_all_batches_experiments(filepath_dict=filepath_dict):
    experiment_df_list = []
    for batchnum, batchname in enumerate(filepath_dict.keys(), start = 4):
        # print(csv_path)
        df = csv2df(filepath_dict[batchname])
        experiment_df_list.append(process_batch(df, '{}'.format(batchnum)))
    
    return pd.concat(experiment_df_list,ignore_index=True)


def main():
    experiments_dataframe = collate_all_batches_experiments(filepath_dict)
    experiments_dataframe.to_pickle("data/mediumrare/unpruned_exps.pkl")


if __name__ == "__main__":
    
    main()





