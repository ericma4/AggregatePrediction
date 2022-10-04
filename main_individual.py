import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
import datetime
from dateutil.relativedelta import relativedelta
from single_run_va_ind import *
import multiprocessing as mp

data = pd.read_feather('/home/jianxin/chars/running/chars_rank_imputed.feather')

predictor_list = []

for i in list(data.columns.values):
    if i.startswith('rank_'):
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
                                                (this_month, data, 72, 48, 1, predictor_list, 'ret', False,))
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
                             predictor_list=predictor_list, target='ret', win=False)
        result = pd.concat([result, Y_table])


# store the result
result = result.reset_index(drop=True)
result.to_feather('anomaly.feather')

best_param = best_param.reset_index(drop=True)
best_param.to_feather('anomaly_param.feather')
