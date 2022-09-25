import numpy as np
import pandas as pd

import datetime
from dateutil.relativedelta import relativedelta

from tuning import  *

# from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import PredefinedSplit,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression, LassoCV, ElasticNetCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error

# from keras.models import Model
# from keras.layers import Dense, Input

import statsmodels.api as sm
import statsmodels.formula.api as smf


def get_data(this_month, data, train_span, valid_span, test_span, predictor_list, target):
    '''

    :param this_month: Prediction start date, should be str and the first day of month like '2022-01-01'
    :param data: Dataframe that contains dependent and independent variables
    :param train_span: Number of months used to train the model
    :param valid_span: Number of months used to validate the model
    :param test_span: Number of months will be predicted, normally it needs to be one to work in this framework
    :param predictor_list: Variables that are used to predict
    :param target: Dependent variable
    :return: X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_train_val, X_train_val
    '''

    # convert the time to MonthEnd
    this_month = pd.to_datetime(this_month)  # Jan 01
    # test_end = this_month + relativedelta(day=31)  # Jan 31

    # split train, validation and test set
    valid_start = this_month + relativedelta(months=-train_span) + relativedelta(day=31)
    train_start = valid_start + relativedelta(months=-valid_span) + relativedelta(day=31)
    test_end = this_month + relativedelta(months=test_span)
    print("\n", 'train_start:', train_start.strftime("%Y-%m-%d"),
          "\n", 'validation_start:', valid_start.strftime("%Y-%m-%d"),
          "\n", 'this_month:', this_month.strftime("%Y-%m-%d"),
          "\n", 'test_end:', test_end.strftime("%Y-%m-%d"))

    # split train, validation and test set
    data_train = data[(data['date'] >= train_start) & (data['date'] < valid_start)]
    data_valid = data[(data['date'] >= valid_start) & (data['date'] < this_month)]
    data_test = data[(data['date'] >= this_month) & (data['date'] < test_end)]

    # split X and Y
    Y_train = data_train[['%s' % target]]
    X_train = data_train[predictor_list]
    Y_val = data_valid[['%s' % target]]
    X_val = data_valid[predictor_list]
    Y_test = data_test[['%s' % target]]
    X_test = data_test[predictor_list]
    Y_train_val = pd.concat([Y_train, Y_val], axis=0)
    X_train_val = pd.concat([X_train, X_val], axis=0)

    # clean X, if some variables have missing value in train and validation set, then drop them from this run
    nan_var = list(X_train.columns[X_train.isnull().any()]) + list(X_val.columns[X_val.isnull().any()])
    X_train = X_train.drop(nan_var, axis=1)
    X_val = X_val.drop(nan_var, axis=1)
    X_test = X_test.drop(nan_var, axis=1)
    X_train_val = X_train_val.drop(nan_var, axis=1)

    print('finish data preprocessing')

    return X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_train_val, X_train_val


def single_run(this_month, data, train_span, valid_span, test_span, predictor_list, target):
    '''

    :param this_month: Prediction start date, should be str and the first day of month like '2022-01-01'
    :param data: Dataframe that contains dependent and independent variables
    :param train_span: Number of months used to train the model
    :param valid_span: Number of months used to validate the model
    :param test_span: Number of months will be predicted, normally it needs to be one to work in this framework
    :param predictor_list: Variables that are used to predict
    :param target: Dependent variable
    :return: Dataframe with four columns: predicted method, predicted value, date and name of dependent variable
    '''
    print("=" * 20, this_month, "=" * 20)

    # get data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_train_val, X_train_val = get_data(this_month, data, train_span, valid_span, test_span, predictor_list,  target)
    # allocate space
    Y_pred_dict = {}

    ####################################################################
    # get result from all other methods

    # 1. individual regression
    for j in X_train_val.columns.values:
        reg = LinearRegression().fit(X_train_val[[j]], Y_train_val)
        Y_pred = reg.predict(X_test[[j]])
        Y_pred_dict['%s' % j] = Y_pred.item(0)

    #### combination method
    df = pd.DataFrame.from_dict(Y_pred_dict, orient='index')
    df.index = X_train_val.columns.values
    # print(df)

    # mean combinations
    Y_pred = df.mean()
    Y_pred_dict['Mean Combination'] = Y_pred.values.item(0)

    # median combinations
    Y_pred = df.median()
    Y_pred_dict['Median Combination'] = Y_pred.values.item(0)

    # 2. ols all
    reg = LinearRegression().fit(X_train_val, Y_train_val)
    Y_pred = reg.predict(X_test)
    Y_pred_dict['OLS'] = Y_pred.item(0)

    # 3. LassoCV for pool
    reg = LassoCV(cv=3, alphas=param_alpha, verbose=0, n_jobs=-1, selection='random', max_iter=10000).fit(X_train_val, Y_train_val.values.ravel())
    Y_pred = reg.predict(X_test)
    Y_pred_dict['Lasso'] = Y_pred.item(0)

    # 4. Multitask ridgeCV
    reg = ElasticNetCV(l1_ratio=0, alphas=param_alpha, cv=3, verbose=0, n_jobs=-1, selection='random', max_iter=10000).fit(X_train_val, Y_train_val.values.ravel())
    Y_pred = reg.predict(X_test)
    Y_pred_dict['Ridge'] = Y_pred.item(0)

    # 5. pca = PCA(n_components= 3)
    # pca transform data
    num_component = 3
    pca = PCA(n_components=num_component)
    X_train_reduced = pca.fit_transform(X_train_val)
    X_test_reduced = pca.transform(X_test)
    # ols
    reg = LinearRegression().fit(X_train_reduced, Y_train_val)
    Y_pred = reg.predict(X_test_reduced)
    Y_pred_dict['PCA'] = Y_pred.item(0)

    # 6. pls = PLS(n_components= 3)
    # PLS transform data
    num_component = 3
    pls = PLSRegression(n_components=num_component)
    # pls fit data
    pls.fit(X_train_val, Y_train_val).predict(X_test)
    Y_pred_dict['PLS'] = Y_pred.item(0)

    # 7. Gradient boost tree regression
    # Using grid search to CV
    # define estimator
    estimator = GradientBoostingRegressor()
    # define parameter grid
    param_grid = {'learning_rate': GBtree_learning_rate,
                  'max_depth': GBtree_max_depth,
                  'n_estimators': GBtree_n_estimators}
    reg = GridSearchCV(estimator=estimator, cv=3, param_grid=param_grid, n_jobs=-1).fit(X_train_val, Y_train_val.values.ravel())
    Y_pred = reg.predict(X_test)
    Y_pred_dict['GBRT'] = Y_pred.item(0)

    # 11. Random Forrest regression
    # Using grid search to CV
    # define estimator
    estimator = RandomForestRegressor()
    # define parameter grid
    param_grid = {'n_estimators': RF_n_estimators,
                  'max_depth': RF_max_depth}
    reg = GridSearchCV(estimator=estimator, cv=3, param_grid=param_grid, n_jobs=-1).fit(X_train_val, Y_train_val.values.ravel())
    Y_pred = reg.predict(X_test)
    Y_pred_dict['RF'] = Y_pred.item(0)

    # Convert the final result to dataframe
    result = pd.DataFrame.from_dict(Y_pred_dict, orient='index')
    result['date'] = pd.to_datetime(this_month) + relativedelta(day=31)
    result['target'] = target
    result = result.reset_index()
    result.columns = ['method', 'y_pred', 'date', 'target']

    return result

