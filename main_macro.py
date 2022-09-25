import os
import numpy as np
import pandas as pd
from pandas.tseries.offsets import *
import datetime
from dateutil.relativedelta import relativedelta
from single_run import *
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import warnings
simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

data = pd.read_stata('Aggregate.dta')
data = data.rename(columns={'yyyy': 'year', 'mm': 'month'})
data['day'] = 1
data['date'] = pd.to_datetime(data[['year', 'month', 'day']]) + MonthEnd(0)

# shift return to predict
data['RF1M'] = data['RF1M'].shift(-1)
data['RF3M'] = data['RF3M'].shift(-3)
data['RF12M'] = data['RF12M'].shift(-12)
data = data[(data['year'] >= 1926) & (data['year'] <= 2021)]
data = data.drop(['year', 'month', 'day'], axis=1)

predictor_list = ['bm', 'corpr', 'de', 'dfr', 'dfy', 'dp',
       'dy', 'ep', 'infl', 'ltr', 'lty', 'mktrx12M',
       'mktrx1M', 'mktrx3M', 'ntis', 'svar',
       'tbl', 'tms', 'MSC', 'equis', 'pdnd', 'ripo', 'nipo', 'cefd',
       'TI_P19', 'TI_P112', 'TI_P29', 'TI_P212', 'TI_P39', 'TI_P312',
       'TI_M9', 'TI_M12', 'TI_V19', 'TI_V112', 'TI_V29', 'TI_V212',
       'TI_V39', 'TI_V312', 'avghrs', 'uic', 'ocons', 'ocap', 'bpt',
       'ipg']

result = pd.DataFrame()

for target in ['RF1M', 'RF3M', 'RF12M']:
       for year in range(1956, 2021):
              for month in range(1, 13):
                     this_month = str(year) + '-' + str(month) + '-01'
                     Y_table = single_run(this_month=this_month, data=data, train_span=240, valid_span=120, test_span=1,
                                          predictor_list=predictor_list, target=target)
                     result = pd.concat([result, Y_table])


result = result.reset_index(drop=True)
result.to_feather('marco_only.feather')