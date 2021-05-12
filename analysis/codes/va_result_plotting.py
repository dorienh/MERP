# %%
import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
log_linear_p = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg-prof/saved_models/experiment_log_linear1.pkl')
log_lstm_p = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg-prof/saved_models/experiment_log_lstm1.pkl')
log_linear = pd.read_pickle('/home/meowyan/Documents/emotion/method-hilang/saved_models/experiment_log3.pkl')
log_lstm = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg/saved_models/experiment_log3.pkl')


# %%
# just getting the labels for the x axis...
profile_list = pd.Series(['-'.join(map(str, l)) for l in log_lstm_p['conditions']])
profile_list = list(profile_list.unique())
profile_list.append('no_profile')

#%%
def get_losses_from_df_p(logdf, startidx, endidx=None):

    def split_list_v_a(valist):
        vlist = []
        alist = []

        for idx, e in enumerate(valist):
            if idx%2 == 0:
                vlist.append(e)
            else:
                alist.append(e)
        return vlist, alist

    mse = logdf['tt_mse'][startidx:endidx].astype(float)
    pcc = logdf['tt_r'][startidx:endidx].astype(float)

    mse_v, mse_a = split_list_v_a(mse)
    pcc_v, pcc_a = split_list_v_a(pcc)
    # print('mse: ', mse.to_list())
    # print('mse v: ', mse_v)
    # print('mse_a: ', mse_a)
    return mse_v, mse_a, pcc_v, pcc_a

mse_v, mse_a, pcc_v, pcc_a = get_losses_from_df_p(log_lstm_p, 2)
lstm_loss = {
    'mse_v': mse_v,
    'mse_a': mse_a,
    'pcc_v': pcc_v,
    'pcc_a': pcc_a
}
mse_v, mse_a, pcc_v, pcc_a = get_losses_from_df_p(log_linear_p, 20)
linear_loss = {
    'mse_v': mse_v,
    'mse_a': mse_a,
    'pcc_v': pcc_v,
    'pcc_a': pcc_a
}
#%%
'''
without profile information
'''
# to be edited accordingly.

def get_losses_from_row_np(logdf, idx_of_interest):
    row_of_interest = logdf.iloc[idx_of_interest]

    noprofile_mse = float(row_of_interest['tt_mse'])
    noprofile_pcc = float(row_of_interest['tt_r'])
    return noprofile_mse, noprofile_pcc

idx_of_interest_a = 23
idx_of_interest_v = 24

np_mse_v, np_pcc_v = get_losses_from_row_np(log_lstm, idx_of_interest_v)
np_mse_a, np_pcc_a = get_losses_from_row_np(log_lstm, idx_of_interest_a)

lstm_loss['mse_v'].append(np_mse_v)
lstm_loss['mse_a'].append(np_mse_a)
lstm_loss['pcc_v'].append(np_pcc_v)
lstm_loss['pcc_a'].append(np_pcc_a)


idx_of_interest_a = 17
idx_of_interest_v = 18

np_mse_v, np_pcc_v = get_losses_from_row_np(log_linear, idx_of_interest_v)
np_mse_a, np_pcc_a = get_losses_from_row_np(log_linear, idx_of_interest_a)

linear_loss['mse_v'].append(np_mse_v)
linear_loss['mse_a'].append(np_mse_a)
linear_loss['pcc_v'].append(np_pcc_v)
linear_loss['pcc_a'].append(np_pcc_a)

# %%
fig = plt.figure(figsize=(6,6))
X = np.arange(0,11) # the 10 profile types. later add 1 for non profile. 
plt.plot(profile_list, lstm_loss['mse_v'], '-o', color='orange', label='lstm')
plt.plot(profile_list, linear_loss['mse_v'], '-o', color='purple', label='linear')
plt.xticks(X, profile_list, rotation='vertical')
plt.ylim([0,0.3])
plt.ylabel('mse - valence')
plt.legend()
plt.tight_layout()
plt.savefig('../plots/va_results/profile_comp_mse_v.png')
plt.show()
plt.close()

# %%
fig = plt.figure(figsize=(6,6))
X = np.arange(0,11) # the 10 profile types. later add 1 for non profile. 
plt.plot(profile_list, lstm_loss['mse_a'], '-o', color='orange', label='lstm')
plt.plot(profile_list, linear_loss['mse_a'], '-o', color='purple', label='linear')
plt.xticks(X, profile_list, rotation='vertical')
plt.ylim([0,0.3])
plt.ylabel('mse - arousal')
plt.legend()
plt.tight_layout()
plt.savefig('../plots/va_results/profile_comp_mse_a.png')
plt.show()
plt.close()

# %%
fig = plt.figure(figsize=(6,6))
X = np.arange(0,11) # the 10 profile types. later add 1 for non profile. 
plt.plot(profile_list, lstm_loss['pcc_v'], '-o', color='orange', label='lstm')
plt.plot(profile_list, linear_loss['pcc_v'], '-o', color='purple', label='linear')
plt.xticks(X, profile_list, rotation='vertical')
plt.ylim([-0.15,0.2])
plt.ylabel('pearson - valence')
plt.legend()
plt.tight_layout()
plt.savefig('../plots/va_results/profile_comp_r_v.png')
plt.show()
plt.close()

# %%
fig = plt.figure(figsize=(6,6))
X = np.arange(0,11) # the 10 profile types. later add 1 for non profile. 
plt.plot(profile_list, lstm_loss['pcc_a'], '-o', color='orange', label='lstm')
plt.plot(profile_list, linear_loss['pcc_a'], '-o', color='purple', label='linear')
plt.xticks(X, profile_list, rotation='vertical')
plt.ylim([-0.15,0.2])
plt.ylabel('pearson - arousal')
plt.legend()
plt.tight_layout()
plt.savefig('../plots/va_results/profile_comp_r_a.png')
plt.show()
plt.close()
# %%

# %%
