import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
import datetime
from dateutil.relativedelta import relativedelta
from single_run_va_ind import *
import multiprocessing as mp
import wrds

###################
# Connect to WRDS #
###################
conn = wrds.Connection()

# CRSP Block
ff = conn.raw_sql("""
                      select rf, date
                      from ff.factors_monthly
                      """)

conn.close()

ff['date'] = pd.to_datetime(ff['date']) + MonthEnd(0)

data = pd.read_feather('chars60_rank_no_impute.feather')
data = data[data['ret'] != 0]

macro = pd.read_csv('marco.csv')
macro['date'] = pd.to_datetime(macro['date'])
macro = macro.drop(['x_month'], axis=1)

data = pd.merge(data, macro, how='left', on='date')
data = pd.merge(data, ff, how='left', on='date')

data['exret'] = data['ret'] - data['rf']

predictor_list = []

for i in list(data.columns.values):
    if i.startswith('rank_') | i.startswith('x_'):
        predictor_list.append(i)
    else:
        pass

result = pd.DataFrame()

param = pd.DataFrame()

##############################################
# Train the models with one process per year #
##############################################

pool = mp.Pool()
p_dict = {}

# train all the models
for year in range(1938, 2021):
    this_month = str(year) + '-' + str(1) + '-01'
    p_dict['p' + str(year)] = pool.apply_async(model_train,
                                                (this_month, data, 72, 48, 1, predictor_list, 'exret', False,))
    print('process %s' % year)

pool.close()
pool.join()

# and then predict
for year in range(1938, 2021):
    model_dict, best_param = p_dict['p%s' % year].get()
    for month in range(1, 13):
        this_month = str(year) + '-' + str(month) + '-01'
        # note that here the train and valid span are not used since we just use test sample in the single_run function here
        Y_table = single_run(model_dict=model_dict, this_month=this_month, data=data, train_span=72,
                             valid_span=48, test_span=1,
                             predictor_list=predictor_list, target='exret', win=False)
        result = pd.concat([result, Y_table])
        param = pd.concat([param, best_param])

# store the result
result = result.reset_index(drop=True)
result.to_feather('r_hat_mlfeather')

param = param.reset_index(drop=True)
param.to_feather('r_hat_ml_param.feather')

# Create a list of methods excluding 'permno', 'y', 'date', 'target'
methods = [col for col in result.columns if col not in ['permno', 'y', 'date', 'target']]

# Initialize an empty DataFrame for storing results
result_oos = pd.DataFrame()

# Initialize an empty dictionary for storing oosr2 values
oosr2_dict = {}

# calculate oosR2
for method in methods:
    temp = result[['%s' % method, 'y' ]]
    oosr2 = 1 - np.sum(np.square(temp['%s' % method] - temp['y']))/np.sum(np.square(0 - temp['y']))
    oosr2_dict[method] = oosr2

# Convert the oosr2 dictionary into a DataFrame and concatenate it with result_oos
oosr2_df = pd.DataFrame.from_dict(oosr2_dict, orient='index', columns=['oosr2'])
result_oos = pd.concat([result_oos, oosr2_df])

result_oos.to_csv('result_oos.csv')
