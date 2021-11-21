'''
references:
https://www.analyticsvidhya.com/blog/2020/06/introduction-anova-statistics-data-science-covid-python/
https://www.reneshbedre.com/blog/anova.html
https://vknight.org/unpeudemath/python/2016/08/13/Analysis-of-variance-with-different-sized-sample.html
https://www.spss-tutorials.com/levenes-test-in-spss/
https://nbviewer.jupyter.org/github/Praveen76/ANOVA-Test-COVID-19/blob/master/One%20Way%20ANOVA%20Test.ipynb
https://www.statsmodels.org/stable/example_formulas.html
'''

# %%
from operator import sub
import pandas as pd
pd.set_option('mode.chained_assignment', None)
# https://www.dataquest.io/blog/settingwithcopywarning/
import numpy as np

import scipy.stats as stats
import os
import random

import statsmodels.api as sm
import statsmodels.stats.multicomp

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
# %%
exps = pd.read_pickle('../../data/exps_ready3.pkl')
pinfo_numero = pd.read_pickle('../../data/pinfo_numero.pkl')
# %%
r_mapper_dict = {'age': {0.0: '(0, 25)', 0.33: '(26, 35)', 0.66: '(36, 50)', 1.0: '(51, 80)'}, 
'gender': {1.0: 'Female', 0.5: 'Other', 0.0: 'Male'}, 
'residence': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'enculturation': {1.0: 'USA', 0.5: 'India', 0.0: 'Other'}, 
'language': {1.0: 'English', 0.5: 'Tamil', 0.0: 'Other'}, 
'genre': {1.0: 'Rock', 0.66: 'Classical', 0.33: 'Pop', 0.0: 'Other'}, 
'instrument': {1.0: 'Yes', 0.0: 'No'}, 
'training': {1.0: 'Yes', 0.0: 'No'}, 
'duration': {0: '(0, 0)', 0.5: '(1, 5)', 1.0: '(6, 50)'}, 
'master': {1.0: 'Yes', 0.0: 'No'}}

pinfo = pinfo_numero.replace(r_mapper_dict)


# %%
affect_type = 'arousals'
profile_type = 'genre'


# %%
# what is the size of the smallest profile group?
def print_group_counts():
    for profile in r_mapper_dict.keys():
        print(profile)
        sub_exps = exps[['workerid',affect_type]]
        mapping = dict(pinfo[['workerid',profile]].values)
        sub_exps['group'] = sub_exps.workerid.map(mapping)
        print(sub_exps.groupby('group').count())
# print_group_counts()
'''
requirement: equal number of observations in each group << NOT TRUE. baka internet.
gender, others: 12 only. 
next smallest is 168 for enculturation, under others. 
I think 100 is a good number, as for gender, remove the single other participant ba. 
but, remove completely from all analysis? or only remove when taking gender into acc? o.o
'''
    
# %%
def get_sub_exps(affect_type, profile_type, exps):
    
    sub_exps = exps[['workerid',affect_type]]
    # average the affect labels for each trial
    sub_exps[affect_type] = sub_exps[affect_type].apply(lambda x: np.mean(x))

    # map workerid to profile group
    mapping = dict(pinfo[['workerid',profile_type]].values)
    sub_exps['group'] = sub_exps.workerid.map(mapping)
    return sub_exps

# %%
def reformat_and_power_transform(sub_exps, affect_type, profile_type):
# power transform
    temp_dict = {}
    for group in r_mapper_dict[profile_type].values():
        temp_dict[group], _ = stats.yeojohnson(sub_exps[sub_exps['group'] == group][affect_type])
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in temp_dict.items() ]))
    return df

# %%
# result_dict = {}
for profile_type in r_mapper_dict.keys():
    # result_dict[profile_type] = {}
    print(f"# profile: {profile_type}")
    for affect_type in ['arousals', 'valences']:
        print(f"## affect type: {affect_type}")
        sub_exps = get_sub_exps(affect_type,profile_type, exps)
        df = reformat_and_power_transform(sub_exps, affect_type, profile_type)
        data = [df[col].dropna() for col in df]
        _, pval = stats.levene(*data)
        if pval>0.05:
            print('homogeneity of variance satisfied.')
        
# data
# %%
F, p = stats.f_oneway(*data)
# Seeing if the overall model is significant
print('F-Statistic=%.3f, p=%.3f' % (F, p))

newDf=df.stack().to_frame().reset_index().rename(columns={'level_1':'group',0:affect_type})
del newDf['level_0']

#Post hoc test
# use tukey because of unequal sample sizes
mc = statsmodels.stats.multicomp.MultiComparison(newDf['arousals'],newDf['group'])
mc_results = mc.tukeyhsd()
print(mc_results)



# check normality (bigger is better)
# small p value suggests evidence that data is not normally distributed.
for group in r_mapper_dict[profile_type].values():
    print(group, stats.shapiro(df[group]))
# Levene variance test (>0.05 is required. since this is not satisfied, forget it. ignore anova.)
# homogeneity of variance
# small p value suggests evidence that variances are not equal.
stats.levene(*data)



# %%

# sub_exps = get_sub_exps('arousals','master', exps)
# df = reformat_and_power_transform(sub_exps, 'arousals', 'master')
# data = [df[col].dropna() for col in df]
def plot_dist(sub_exps, profile_type, affect_type):
    # before power transform
    for group in r_mapper_dict[profile_type].values():
        subset = sub_exps[sub_exps['group'] == group]
        sns.distplot(subset[affect_type],label=group)
    plt.legend()
    plt.show()

def plot_dist_av(profile_type):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
    # print(axes[1])
    for idx, affect_type in enumerate(['arousals', 'valences']):
        sub_exps = get_sub_exps(affect_type, profile_type, exps)
        for group in r_mapper_dict[profile_type].values():
            subset = sub_exps[sub_exps['group'] == group]
            sns.distplot(subset[affect_type],label=group, ax=axes[idx])
        
    handles, labels = axes[idx].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(profile_type)
    plt.show()

    # after power transform 
    # for group in r_mapper_dict[profile_type].values():
    #     # sns.distplot(df[group])
    #     # sns.distplot(subset[affect_type],label=group)
    #     subset = sub_exps[sub_exps['group'] == group]
    #     subset[affect_type],fitted_lambda = stats.yeojohnson(subset[affect_type])
    #     sns.distplot(subset[affect_type],label=group)
        
# plot_dist()

for profile_type in r_mapper_dict.keys():
    
    print(f"# profile: {profile_type}")
    for affect_type in ['arousals', 'valences']:
        plot_dist_av(profile_type)
        #plot
        # plot_dist(sub_exps, profile_type, affect_type)
    break
# %%
# reformat dataframe for anova
def reformat_df():
    temp_dict = {}
    for group in r_mapper_dict[profile_type].values():
        temp_dict[group] = list(sub_exps[sub_exps['group']==group][affect_type])
    df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in temp_dict.items() ]))
    # df.describe()
    return df

# %%
def random_select_samples(exps, num_samples=100):
    np.random.seed(1234)
    temp_dict = {}
    for group in r_mapper_dict[profile_type].values():
        temp_dict[group] = random.sample(list(exps[affect_type][exps['group']==group]), num_samples)
    dataNew=pd.DataFrame(temp_dict)
    return dataNew


# %%
# Ordinary Least Squares (OLS) model
# del sub_exps['workerid']
newDf=df.stack().to_frame().reset_index().rename(columns={'level_1':'group',0:affect_type})
del newDf['level_0']
model = ols(f'{affect_type} ~ C(group)', data=newDf).fit()
model.summary()
# %%
anova_table = sm.stats.anova_lm(model, typ=2)
anova_table
# %%
#Post hoc test
# use tukey because of unequal sample sizes
mc = statsmodels.stats.multicomp.MultiComparison(newDf['valences'],newDf['group'])
mc_results = mc.tukeyhsd()
print(mc_results)
# %%
from bioinfokit.analys import stat
res = stat()
res.tukey_hsd(df=newDf, res_var='arousals', xfac_var='group', anova_model='arousals ~ C(group)')
res.tukey_summary
# %%
sm.qqplot(res.anova_std_residuals, line='45')
plt.xlabel("Theoretical Quantiles")
plt.ylabel("Standardized Residuals")
plt.show()

res = model.resid 
fig = sm.qqplot(res, line='s')
plt.show()


# histogram
plt.hist(res.anova_model_out.resid, bins='auto', histtype='bar', ec='k') 
plt.xlabel("Residuals")
plt.ylabel('Frequency')
plt.show()


# %%
# %%
# the plots look normally distributed but the qqplots show they are NOT.
fig = plt.figure(figsize=(10,10))
title = fig.suptitle("Arousal across different profile groups", fontsize=14)
fig.subplots_adjust(top=0.85, wspace=0.3)

ax1 = fig.add_subplot(2,2,1)
sns.kdeplot(dataNew['(0, 25)'], ax=ax1, shade=True,bw=4, color='g')

ax2 = fig.add_subplot(2,2,2)
sns.kdeplot(dataNew['(26, 35)'], ax=ax2, shade=True,bw=4, color='y')

ax2 = fig.add_subplot(2,2,3)
sns.kdeplot(dataNew['(36, 50)'], ax=ax2, shade=True,bw=4, color='r')

ax2 = fig.add_subplot(2,2,4)
sns.kdeplot(dataNew['(51, 80)'], ax=ax2, shade=True,bw=4, color='b')
# %%
sm.qqplot(dataNew['(0, 25)'], line ='45') 
sm.qqplot(dataNew['(26, 35)'], line ='45') 
sm.qqplot(dataNew['(36, 50)'], line ='45') 
sm.qqplot(dataNew['(51, 80)'], line ='45') 
# %%
# check normality
# small p value suggests evidence that data is not normally distributed.
for group in r_mapper_dict[profile_type].values():
    print(stats.shapiro(dataNew[group]))
# %%
# Levene variance test  
# homogeneity of variance
# small p value suggests evidence that variances are not equal.
stats.levene(dataNew['(0, 25)'],dataNew['(26, 35)'],dataNew['(36, 50)'],dataNew['(51, 80)'])
# %%
