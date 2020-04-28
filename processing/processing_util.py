'''
Shared variables and functions in processing folder
'''
import pandas as pd

batch_names = {'four':'4', 'five':'5', 'six':'6', 'seven':'7', 'eight':'8'}

filepath_dict = {
    'four' : "data/results_mTurk/batch_4_US_master.csv",
    'five' : "data/results_mTurk/Batch_5_US_master.csv",
    'six'  : "data/results_mTurk/Batch6_worldwide_master.csv",
    'seven': "data/results_mTurk/batch7_all_world_NONmaster.csv",
    'eight': "data/results_mTurk/batch8_all_world_NONmaster.csv"
}

def csv2df(filepath):
    df = pd.read_csv(filepath)
    print('shape of df: ', df.shape)
    return df
