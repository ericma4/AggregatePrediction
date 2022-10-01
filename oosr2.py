import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
import datetime

# read macro data and creat benchmark for comparison
macro = pd.read_excel('Aggregate.xlsx')
macro = macro.rename(columns={'yyyy': 'year', 'mm': 'month'})
macro['day'] = 1
macro['date'] = pd.to_datetime(macro[['year', 'month', 'day']]) + MonthEnd(0)

macro['Rx1M'] = macro['Rx1M'].shift(-1)
macro['Rx3M'] = macro['Rx3M'].shift(-3)
macro['Rx12M'] = macro['Rx12M'].shift(-12)
macro['dg_r'] = macro['dg_r'].shift(-12)

macro['Rx1M_rolling'] = macro.rolling(window=360)['Rx1M'].mean()
macro['Rx3M_rolling'] = macro.rolling(window=360)['Rx3M'].mean()
macro['Rx12M_rolling'] = macro.rolling(window=360)['Rx12M'].mean()
macro['dg_r_rolling'] = macro.rolling(window=360)['dg_r'].mean()

# read predicted result
data = pd.read_feather('macro_only.feather')
data = pd.merge(data, macro[['date', 'Rx1M_rolling', 'Rx3M_rolling', 'Rx12M_rolling', 'dg_r_rolling', 'Rx1M', 'Rx3M', 'Rx12M', 'dg_r']], how='left', on=['date'])

result = pd.DataFrame()

# calculate oosR2
for target in data['target'].unique():
    oosr2 = {}
    for method in data['method'].unique():
        temp = data[(data['method'] == '%s' % method) & (data['target'] == '%s' % target)]
        oosr2['%s' % method] = 1 - np.sum(np.square(temp['y_pred'] - temp['%s' % target]))/np.sum(np.square(temp['%s_rolling' % target] - temp['%s' % target]))
    oosr2_df = pd.DataFrame.from_dict(oosr2, orient='index')
    oosr2_df['target'] = '%s' % target
    result = pd.concat([result, oosr2_df])

result = result.reset_index()
result.columns = ['method', 'oosr2', 'target']

result = result.pivot(index='target', columns='method', values='oosr2').reset_index()
result = result[['target', 'GBRT', 'Lasso', 'Mean Combination',
       'Median Combination', 'OLS', 'PCA', 'PLS', 'RF', 'Ridge']]

result.to_csv('macro_only_oosr2.csv', index=0)