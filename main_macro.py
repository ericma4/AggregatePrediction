import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
import datetime
from dateutil.relativedelta import relativedelta
from single_run_va import *
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import warnings
simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_excel('Aggregate.xlsx')
data = data.rename(columns={'yyyy': 'year', 'mm': 'month'})
data['day'] = 1
data['date'] = pd.to_datetime(data[['year', 'month', 'day']]) + MonthEnd(0)

# shift return to predict
data['Rx1M'] = data['Rx1M'].shift(-1)
data['Rx3M'] = data['Rx3M'].shift(-3)
data['Rx12M'] = data['Rx12M'].shift(-12)
data['dg_r'] = data['dg_r'].shift(-12)
data = data[(data['year'] >= 1926) & (data['year'] <= 2021)]
data = data.drop(['year', 'month', 'day'], axis=1)

predictor_list = ['bm', 'corpr', 'de', 'dfr', 'dfy', 'dp', 'dy', 'ep',
       'infl', 'lRfree', 'ltr', 'lty', 'ntis', 'svar', 'tbl', 'tms',
       'MSC', 'equis', 'pdnd', 'ripo', 'nipo', 'cefd', 'TI_P19',
       'TI_P112', 'TI_P29', 'TI_P212', 'TI_P39', 'TI_P312', 'TI_M9',
       'TI_M12', 'TI_V19', 'TI_V112', 'TI_V29', 'TI_V212', 'TI_V39',
       'TI_V312', 'avghrs', 'uic', 'uemp', 'ocons', 'ocap', 'bpt', 'ipg']

result_Rx1M = pd.DataFrame()
result_Rx3M = pd.DataFrame()
result_Rx12M = pd.DataFrame()
result_dg_r = pd.DataFrame()

param_Rx1M = pd.DataFrame()
param_Rx3M = pd.DataFrame()
param_Rx12M = pd.DataFrame()
param_dg_r = pd.DataFrame()

for year in range(1958, 2021):
       for month in range(1, 13):
              this_month = str(year) + '-' + str(month) + '-01'
              # Rx1M
              Y_table, best_param = single_run(this_month=this_month, data=data, train_span=240, valid_span=120, test_span=1,
                                   predictor_list=predictor_list, target='Rx1M')
              result_Rx1M = pd.concat([result_Rx1M, Y_table])
              param_Rx1M = pd.concat([param_Rx1M, best_param])
              # Rx3M
              Y_table, best_param = single_run(this_month=this_month, data=data, train_span=240, valid_span=120, test_span=3,
                                   predictor_list=predictor_list, target='Rx3M')
              result_Rx3M = pd.concat([result_Rx3M, Y_table])
              param_Rx3M = pd.concat([param_Rx3M, best_param])
              # Rx12M
              Y_table, best_param = single_run(this_month=this_month, data=data, train_span=240, valid_span=120, test_span=12,
                                   predictor_list=predictor_list, target='Rx12M')
              result_Rx12M = pd.concat([result_Rx12M, Y_table])
              param_Rx12M = pd.concat([param_Rx12M, best_param])
              # dg_r
              Y_table, best_param = single_run(this_month=this_month, data=data, train_span=240, valid_span=120, test_span=12,
                                   predictor_list=predictor_list, target='dg_r')
              result_dg_r = pd.concat([result_dg_r, Y_table])
              param_dg_r = pd.concat([param_dg_r, best_param])


result = pd.concat([result_Rx1M, result_Rx3M, result_Rx12M, result_dg_r])
result = result.reset_index(drop=True)
result.to_feather('macro_only.feather')

best_param = pd.concat([param_Rx1M, param_Rx3M, param_Rx12M, param_dg_r])
best_param = best_param.reset_index(drop=True)
best_param.to_feather('macro_param.feather')