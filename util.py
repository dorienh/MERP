'''
global variables and functions
'''

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