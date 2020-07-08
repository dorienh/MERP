'''
global variables and functions
'''
import numpy as np

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

#########################################
########    Load Pickle File    #########
#########################################
import pickle

def load_pickle(filepath):
    # path = os.path.join(os.path.abspath('..'), 'data', filename)
    with open(filepath, 'rb') as handle:
        unpickled = pickle.load(handle)
    return unpickled


#########################################
########       Song List        #########
#########################################

songlist = ['deam_115', 'deam_343', 'deam_745', 'deam_1334', \
    '00_35', '00_145', '00_275', '00_366', '00_661', '00_695', '00_702', '00_839', '00_882', '00_883', \
    '01_71', '01_139', '01_143', '01_154', '01_175', '01_177', '01_228', '01_333', '01_897', '01_890', \
    '0505_33', '0505_46', '0505_58', '0505_62', '0505_80', '0505_85', '0505_90', '0505_102', '0505_184', '0505_199', \
    '10_93', '10_130', '10_150', '10_243', '10_288', '10_404', '10_487', '10_828', '10_942', '10_216', \
    '11_209', '11_411', '11_427', '11_459', '11_505', '11_528', '11_648', '11_693', '11_748',  '11_801']


#########################################
########       Train List        ########
#########################################

trainlist = ['deam_115', 'deam_343', 'deam_745', 'deam_1334', \
    '00_366', '00_661', '00_695', '00_702', '00_839', '00_882', '00_883', \
    '01_154', '01_175', '01_177', '01_228', '01_333', '01_897', '01_890', \
    '0505_62', '0505_80', '0505_85', '0505_90', '0505_102', '0505_184', '0505_199', \
    '10_243', '10_288', '10_404', '10_487', '10_828', '10_942', '10_216', \
    '11_459', '11_505', '11_528', '11_648', '11_693', '11_748',  '11_801']

#########################################
########       Test List        #########
#########################################

testlist = ['00_35', '00_145', '00_275', \
    '01_71', '01_139', '01_143', \
    '0505_33', '0505_46', '0505_58', \
    '10_93', '10_130', '10_150', \
    '11_209', '11_411', '11_427']

#########################################
#######       Label Types        ########
#########################################

labeltypes = ['arousals', 'valences']