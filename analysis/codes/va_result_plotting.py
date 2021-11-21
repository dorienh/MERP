# %%
import os
import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('../..'))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
master = False
# !!!!! WARNING !!!!!! Line indexes might need to be MANUALLY CHANGED.
if not master:
    log_lstm_p = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg-prof/saved_models/test_log_lstm.pkl')
    log_linear_p = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg-prof/saved_models/test_log_linear.pkl')
    log_lstm = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg/saved_models/test_log_lstm.pkl')
    log_linear = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg/saved_models/test_log_linear.pkl')
    
else:
    log_lstm_p = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg-prof/saved_models_m/test_log_lstm.pkl')
    log_linear_p = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg-prof/saved_models_m/test_log_linear.pkl')
    log_lstm = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg/saved_models_m/test_log_lstm.pkl')
    log_linear = pd.read_pickle('/home/meowyan/Documents/emotion/method-rdmseg/saved_models_m/test_log_linear.pkl')
# %%
# just getting the labels for the x axis...
profile_list = pd.Series(['-'.join(map(str, l)) for l in log_lstm_p['conditions']])
profile_list = list(profile_list.unique())
profile_list.append('no_profile')
# print(profile_list)
profile_list = ['age', 'gender', 'residence', 'enculturation', 'language', 'genre', 'instrument', 'training', 'duration', 'master', 'no profile']

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
if not master:
    mse_v, mse_a, pcc_v, pcc_a = get_losses_from_df_p(log_lstm_p, 0) # previously 2
else:
    mse_v, mse_a, pcc_v, pcc_a = get_losses_from_df_p(log_lstm_p, 1) # master
lstm_loss = {
    'mse_v': mse_v,
    'mse_a': mse_a,
    'pcc_v': pcc_v,
    'pcc_a': pcc_a
}
if not master:
    mse_v, mse_a, pcc_v, pcc_a = get_losses_from_df_p(log_linear_p, 0) # previously 44
else:
    mse_v, mse_a, pcc_v, pcc_a = get_losses_from_df_p(log_linear_p, 7) # master
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


if master:
    idx_of_interest_v = 0 
    idx_of_interest_a = 1 
else:
    idx_of_interest_v = 2 # 24
    idx_of_interest_a = 3 # 23

np_mse_v, np_pcc_v = get_losses_from_row_np(log_lstm, idx_of_interest_v)
np_mse_a, np_pcc_a = get_losses_from_row_np(log_lstm, idx_of_interest_a)

lstm_loss['mse_v'].append(np_mse_v)
lstm_loss['mse_a'].append(np_mse_a)
lstm_loss['pcc_v'].append(np_pcc_v)
lstm_loss['pcc_a'].append(np_pcc_a)

if master:
    idx_of_interest_v = 0 
    idx_of_interest_a = 1 
else:
    idx_of_interest_v = 2 # 18
    idx_of_interest_a = 3 # 17

np_mse_v, np_pcc_v = get_losses_from_row_np(log_linear, idx_of_interest_v)
np_mse_a, np_pcc_a = get_losses_from_row_np(log_linear, idx_of_interest_a)

linear_loss['mse_v'].append(np_mse_v)
linear_loss['mse_a'].append(np_mse_a)
linear_loss['pcc_v'].append(np_pcc_v)
linear_loss['pcc_a'].append(np_pcc_a)

# %%
'''
all 4 loss plots in one fig
'''
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10,9))
X = np.arange(0,11) 
ax1.plot(profile_list, lstm_loss['mse_v'], '-o', color='orange', label='lstm')
ax1.plot(profile_list, linear_loss['mse_v'], '-o', color='purple', label='fc')
ax1.set_xticklabels(profile_list, rotation=45, ha='right')
ax1.set_ylim([0,0.35])
ax1.set_ylabel('mse - valence', fontdict=dict(weight='bold'))
ax1.legend()

ax2.plot(profile_list, lstm_loss['mse_a'], '-o', color='orange', label='lstm')
ax2.plot(profile_list, linear_loss['mse_a'], '-o', color='purple', label='fc')
ax2.set_xticklabels(profile_list, rotation=45, ha='right')
ax2.set_ylim([0,0.35])
ax2.set_ylabel('mse - arousal', fontdict=dict(weight='bold'))
ax2.legend()

ax3.plot(profile_list, lstm_loss['pcc_v'], '-o', color='orange', label='lstm')
ax3.plot(profile_list, linear_loss['pcc_v'], '-o', color='purple', label='fc')
ax3.set_xticklabels(profile_list, rotation=45, ha='right')
ax3.set_ylim([-0.15,0.5])
ax3.set_ylabel('pearson - valence', fontdict=dict(weight='bold'))
ax3.legend()

ax4.plot(profile_list, lstm_loss['pcc_a'], '-o', color='orange', label='lstm')
ax4.plot(profile_list, linear_loss['pcc_a'], '-o', color='purple', label='fc')
ax4.set_xticklabels(profile_list, rotation=45, ha='right')
ax4.set_ylim([-0.15,0.5])
ax4.set_ylabel('pearson - arousal', fontdict=dict(weight='bold'))
ax4.legend()

fig.tight_layout(pad=1.2, h_pad=0.2)
fig.subplots_adjust(top=0.96)

if master:
    fig.suptitle('master participants', y=0.99,fontweight='bold',fontsize=12)
    plt.savefig('../plots/va_results/master_loss_comp_p.png')
else:
    fig.suptitle('all participants', y=0.99,fontweight='bold',fontsize=12)
    plt.savefig('../plots/va_results/loss_comp_p.png')


# %%

# %%
# put test losses into nice columns to save as csv or even text file and copy paste to overleaf.

# fc-v-mse, fc-v-pcc, fc-a-mse, fc-a-pcc, lstm-v-mse, lstm-a-mse, lstm-v-pcc, lstm-a-pcc

temp_dict = {
    'profile': profile_list,
    'fc-v-mse': linear_loss['mse_v'],
    'fc-v-pcc': linear_loss['pcc_v'],
    'fc-a-mse': linear_loss['mse_a'],
    'fc-a-pcc': linear_loss['pcc_a'],
    'lstm-v-mse': lstm_loss['mse_v'],
    'lstm-v-pcc': lstm_loss['pcc_v'],
    'lstm-a-mse': lstm_loss['mse_a'],
    'lstm-a-pcc': lstm_loss['pcc_a'],
}


df = pd.DataFrame(temp_dict)
df.to_csv('../plots/va_results/test_results.csv')

# %%
