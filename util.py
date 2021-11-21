'''
global variables and functions
'''
import numpy as np

mapper_dict = {
    'age': {(0,25):0.0, (26,35):0.33, (36,50):0.66, (51,80):1.0},
    'gender': {'Female': 1.0, 'Other': 0.5, 'Male': 0.0},
    'residence': {'US': 1.0, 'IN': 0.5, 'Other': 0.0},
    'enculturation': {'US': 1.0, 'IN': 0.5, 'Other': 0.0},
    'language': {'EN': 1.0, 'TA': 0.5, 'Other': 0.0},
    'genre': {'Rock': 1.0, 'Classical music': 0.66, 'Pop': 0.33, 'Other': 0.0},
    'instrument': {'Yes': 1.0, 'No': 0.0},
    'training': {'Yes': 1.0, 'No': 0.0},
    'duration': {(0,0):0, (1,5):0.5, (6,50):1.0},
    'master': {'Yes': 1.0, 'No': 0.0}
}

# reversed in va_by_profile.py
# not sure if i'll use it but might be useful for plotting output. 
r_mapper_dict = {'age': {0.0: (0, 25), 0.33: (26, 35), 0.66: (36, 50), 1.0: (51, 80)}, 
'gender': {1.0: 'Female', 0.5: 'Other', 0.0: 'Male'}, 
'residence': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'enculturation': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'language': {1.0: 'English', 0.5: 'Tamil', 0.0: 'Other'}, 
'genre': {1.0: 'Rock', 0.66: 'Classical', 0.33: 'Pop', 0.0: 'Other'}, 
'instrument': {1.0: 'Yes', 0.0: 'No'}, 
'training': {1.0: 'Yes', 0.0: 'No'}, 
'duration': {0: (0, 0), 0.5: (1, 5), 1.0: (6, 50)}, 
'master': {1.0: 'Yes', 0.0: 'No'}}

#########################################
############    Smoothing    ############
#########################################

# signal processing variables

order = 6,
fs = 30.0,              # sample rate, Hz
cutoff = 3.667,         # desired cutoff frequency of the filter, Hz
savgol_winlen = 71,     # for ground truth, 21 is enough, but for classification output, using 171
savgol_polyorder = 1,   # tried 0 and 1 and 2, 1 gave the best mae, mse, pearson.


def butter_lowpass():
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(self, data):
    b, a = butter_lowpass()
    y = lfilter(b, a, data)
    return y

def smoothing(self, data, winlen=None):
    if winlen is None:
        winlen = savgol_winlen
    return savgol_filter(data, winlen, savgol_polyorder, mode='nearest') #originally mode='interp', but interp changes the range of the values. 

def cl_output_smoothing(self, data, winlen=None):
    data = butter_lowpass_filter(data)
    return smoothing(data, winlen)

#########################################
########       Normalizing      #########
#########################################

def normalize(data, minimum, maximum):
    ''' function to normalize the values of an array to [-1, 1] given min and max'''
    data = np.array(data)
    return ((data - minimum)/(maximum - minimum + 1e-05))*2 - 1

def normalize_01(data, minimum, maximum):
    ''' function to normalize the values of an array to [0, 1] given min and max'''
    data = np.array(data)
    return ((data - minimum)/(maximum - minimum + 1e-05))

#########################################
########    Load Pickle File    #########
#########################################
import pickle

def load_pickle(filepath):
    # path = os.path.join(os.path.abspath('..'), 'data', filename)
    with open(filepath, 'rb') as handle:
        unpickled = pickle.load(handle)
    return unpickled

def save_pickle(filepath, item):
    with open(filepath, 'wb') as handle:
        pickle.dump(item, handle, protocol=pickle.HIGHEST_PROTOCOL)


#########################################
#####    Count Trainable Params    ######
#########################################

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#########################################
########       Song List        #########
#########################################

songlist = ['deam_115', 'deam_343', 'deam_745', 'deam_1334', \
    '00_35', '00_145', '00_275', '00_366', '00_661', '00_695', '00_702', '00_839', '00_882', '00_883', \
    '01_71', '01_139', '01_143', '01_154', '01_175', '01_177', '01_228', '01_333', '01_897', '01_890', \
    '0505_33', '0505_46', '0505_58', '0505_62', '0505_80', '0505_85', '0505_90', '0505_102', '0505_184', '0505_199', \
    '10_93', '10_130', '10_150', '10_243', '10_288', '10_404', '10_487', '10_828', '10_942', '10_216', \
    '11_209', '11_411', '11_427', '11_459', '11_505', '11_528', '11_648', '11_693', '11_748',  '11_801']

songdict = {
    'deam': ['deam_115', 'deam_343', 'deam_745', 'deam_1334'],
    '00': ['00_35', '00_145', '00_275', '00_366', '00_661', '00_695', '00_702', '00_839', '00_882', '00_883'],
    '01': ['01_71', '01_139', '01_143', '01_154', '01_175', '01_177', '01_228', '01_333', '01_897', '01_890'],
    '0505': ['0505_33', '0505_46', '0505_58', '0505_62', '0505_80', '0505_85', '0505_90', '0505_102', '0505_184', '0505_199'],
    '10': ['10_93', '10_130', '10_150', '10_243', '10_288', '10_404', '10_487', '10_828', '10_942', '10_216'],
    '11': ['11_209', '11_411', '11_427', '11_459', '11_505', '11_528', '11_648', '11_693', '11_748',  '11_801']
}

# #########################################
# ########       Train List        ########
# #########################################

trainlist = ['deam_115', 'deam_343', 'deam_745', 'deam_1334', \
    '00_366', '00_661', '00_695', '00_702', '00_839', '00_882', '00_883', \
    '01_154', '01_175', '01_177', '01_228', '01_333', '01_897', '01_890', \
    '0505_62', '0505_80', '0505_85', '0505_90', '0505_102', '0505_184', '0505_199', \
    '10_243', '10_288', '10_404', '10_487', '10_828', '10_942', '10_216', \
    '11_459', '11_505', '11_528', '11_648', '11_693', '11_748',  '11_801']

# #########################################
# ########       Test List        #########
# #########################################

testlist = ['00_35', '00_145', '00_275', \
    '01_71', '01_139', '01_143', \
    '0505_33', '0505_46', '0505_58', \
    '10_93', '10_130', '10_150', \
    '11_209', '11_411', '11_427']


#########################################
########       Train List        ########
#########################################

# trainlist = ['deam_115', 'deam_343', 'deam_745', 'deam_1334', \
    # '00_366', '00_661', '00_695', '00_702', '00_839', '00_882', \
    # '01_154', '01_175', '01_177', '01_228', '01_333', '01_897', \
    # '0505_62', '0505_80', '0505_85', '0505_90', '0505_102', '0505_184', \
    # '10_243', '10_288', '10_404', '10_487', '10_828', '10_942', \
    # '11_459', '11_505', '11_528', '11_648', '11_693', '11_748']

#########################################
#####       Validation List        ######
#########################################

# validlist = ['00_883', '00_275', \
#     '01_143', '01_890', \
#     '0505_58', '0505_199', \
#     '10_150', '10_216', \
#     '11_801', '11_427']


#########################################
########       Test List        #########
#########################################

# testlist = ['00_35', '00_145', \
#     '01_71', '01_139', \
#     '0505_33', '0505_46',\
#     '10_93', '10_130', \
#     '11_209', '11_411']

#########################################
#######       Label Types        ########
#########################################

labeltypes = ['arousals', 'valences']

#########################################
#### 10 fold cross validation groups ####
#########################################

# group1 = [songdict['00'][0], songdict['01'][0], songdict['0505'][0], songdict['10'][0], songdict['11'][0]]
# group2 = [songdict['00'][1], songdict['01'][1], songdict['0505'][1], songdict['10'][1], songdict['11'][1]]
# group3 = [songdict['00'][2], songdict['01'][2], songdict['0505'][2], songdict['10'][2], songdict['11'][2]]
# group4 = [songdict['00'][3], songdict['01'][3], songdict['0505'][3], songdict['10'][3], songdict['11'][3]]
# group5 = [songdict['00'][4], songdict['01'][4], songdict['0505'][4], songdict['10'][4], songdict['11'][4]]
# group6 = [songdict['00'][5], songdict['01'][5], songdict['0505'][5], songdict['10'][5], songdict['11'][5]]
# group7 = [songdict['00'][6], songdict['01'][6], songdict['0505'][6], songdict['10'][6], songdict['11'][6]]
# group8 = [songdict['00'][7], songdict['01'][7], songdict['0505'][7], songdict['10'][7], songdict['11'][7]]
# group9 = [songdict['00'][8], songdict['01'][8], songdict['0505'][8], songdict['10'][8], songdict['11'][8]]
# group10 = [songdict['00'][9], songdict['01'][9], songdict['0505'][9], songdict['10'][9], songdict['11'][9]]

'''
hardcoded the folds. first 4 have the deam songs, last one doesn't. 

'''
folds = {}
for i in range(0,10,2):
    fold = [songdict['00'][i], songdict['00'][i+1],
            songdict['01'][i], songdict['01'][i+1],
            songdict['0505'][i], songdict['0505'][i+1],
            songdict['10'][i], songdict['10'][i+1],
            songdict['11'][i], songdict['11'][i+1]]
    if i<8:
        fold.append(songdict['deam'][i//2])
    folds[i//2] = fold
print('meow', folds)