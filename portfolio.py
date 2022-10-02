import datetime
import pandas as pd
import numpy as np

chars = pd.read_feather('/home/jianxin/chars/running/chars_rank_imputed.feather')


def chars_bucket(row):
    if row['chars_break'] <= -0.8:
        value = 1
    elif row['chars_break'] <= -0.6:
        value = 2
    elif row['chars_break'] <= -0.4:
        value = 3
    elif row['chars_break'] <= -0.2:
        value = 4
    elif row['chars_break'] <= 0:
        value = 5
    elif row['chars_break'] <= 0.2:
        value = 6
    elif row['chars_break'] <= 0.4:
        value = 7
    elif row['chars_break'] <= 0.6:
        value = 8
    elif row['chars_break'] <= 0.8:
        value = 9
    elif row['chars_break'] > 0.8:
        value = 10
    else:
        value = ''
    return value


# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan


vwret = pd.DataFrame()
var_list = []

for i in list(chars.columns.values):
    if i.startswith('rank_'):
        var_list.append(i)
    else:
        pass

# construct percentile and value-weighted return of different accounting variables
for variable in var_list:
    print('='*20, 'Processing %s' % variable, '='*20)
    chars['chars_break'] = chars['%s' % variable]
    # merge back the percentiles
    temp = chars.copy()
    temp['charport'] = temp.apply(chars_bucket, axis=1)
    # value-weigthed return
    vwret_temp = temp.groupby(['date', 'charport']).apply(wavg, 'ret', 'lag_me').to_frame().reset_index().rename(columns={0: 'vwret'})
    vwret_temp['variable'] = '%s' % variable
    vwret = pd.concat([vwret, vwret_temp])

vwret = vwret.reset_index(drop=True)
vwret.to_feather('vwret_all.feather')

# construct long-short portfolio
vwret = vwret[(vwret['charport'] == 1) | (vwret['charport'] == 10)]
vwret = vwret.pivot(index=['date', 'variable'], columns='charport', values='vwret').reset_index()
vwret['ls'] = vwret[10] - vwret[1]

# specify the sign of long-short portfolio
df_sign = vwret[vwret['date'] <= datetime.datetime(year=2000, month=1, day=1)]
df_sign = df_sign.groupby(['variable'], as_index=False)['ls'].mean()
df_sign['sign'] = np.sign(df_sign['ls'])

vwret = pd.merge(vwret, df_sign[['variable', 'sign']], how='left', on=['variable'])
vwret['ls'] = vwret['ls'] * vwret['sign']

vwret = vwret[['date', 'variable', 'ls']]
vwret = vwret.pivot(index=['date'], columns='variable', values='ls').reset_index()
vwret = vwret.reset_index(drop=True)
vwret.to_feather('vwret_ls.feather')
