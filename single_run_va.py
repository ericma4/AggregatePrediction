import numpy as np
import pandas as pd
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import PredefinedSplit,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import *
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tuning import  *


def get_data(this_month, data, train_span, valid_span, test_span, predictor_list, target):
    '''

    :param this_month: Prediction start date, should be str and the first day of month like '2022-01-01'
    :param data: Dataframe that contains dependent and independent variables
    :param train_span: Number of months used to train the model
    :param valid_span: Number of months used to validate the model
    :param test_span: Numbers of months that dependent variable lagged
    :param predictor_list: Variables that are used to predict
    :param target: Dependent variable
    :return: X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_train_val, X_train_val
    '''

    # convert the time to MonthEnd
    this_month = pd.to_datetime(this_month)  # Jan 01
    # test_end = this_month + relativedelta(day=31)  # Jan 31

    # split train, validation and test set
    test_start = this_month - relativedelta(months=test_span)
    test_end = test_start + relativedelta(months=1)
    valid_start = test_start - relativedelta(months=valid_span)
    train_start = valid_start - relativedelta(months=train_span)
    print("\n", 'train_span:', train_start.strftime("%Y-%m-%d"), '-', (valid_start-relativedelta(days=1)).strftime("%Y-%m-%d"),
          "\n", 'validation_span:', valid_start.strftime("%Y-%m-%d"), '-', (test_start-relativedelta(days=1)).strftime("%Y-%m-%d"),
          "\n", 'test_span:', test_start.strftime("%Y-%m-%d"), '-', (test_end-relativedelta(days=1)).strftime("%Y-%m-%d")
          )

    # split train, validation and test set
    data_train = data[(data['date'] >= train_start) & (data['date'] < valid_start)]
    data_valid = data[(data['date'] >= valid_start) & (data['date'] < test_start)]
    data_test = data[(data['date'] >= test_start) & (data['date'] < test_end)]

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
    nan_var = list(X_train.columns[X_train.isnull().any()]) + list(X_val.columns[X_val.isnull().any()]) + list(X_test.columns[X_test.isnull().any()])
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
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_train_val, X_train_val = get_data(this_month, data, train_span, valid_span, test_span, predictor_list, target)
    # allocate space
    Y_pred_dict = {}
    best_param_dict = {}

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

    # 3. Lasso
    param_dict = {}
    for alpha in param_alpha:
        param_dict[alpha] = Lasso(alpha=alpha).fit(X_train, Y_train.values.ravel()).score(X_val, Y_val)

    best_score = max(param_dict.values())
    best_param = list(param_dict.keys())[list(param_dict.values()).index(best_score)]
    reg = Lasso(alpha=best_param).fit(X_train_val, Y_train_val.values.ravel())
    # refit
    Y_pred = reg.predict(X_test)
    Y_pred_dict['Lasso'] = Y_pred.item(0)
    best_param_dict['Lasso'] = best_param

    # 4. Ridge
    param_dict = {}
    for alpha in param_alpha:
        param_dict[alpha] = Ridge(alpha=alpha).fit(X_train, Y_train.values.ravel()).score(X_val, Y_val)

    best_score = max(param_dict.values())
    best_param = list(param_dict.keys())[list(param_dict.values()).index(best_score)]
    reg = Ridge(alpha=best_param).fit(X_train_val, Y_train_val.values.ravel())

    Y_pred = reg.predict(X_test)
    Y_pred_dict['Ridge'] = Y_pred.item(0)
    best_param_dict['Ridge'] = best_param

    # 5. PCA
    param_dict = {}
    for n in num_component_grid:
        pca = PCA(n_components=n)
        X_train_reduced = pca.fit_transform(X_train)
        X_val_reduced = pca.transform(X_val)
        param_dict[n] = LinearRegression().fit(X_train_reduced, Y_train).score(X_val_reduced, Y_val)

    best_score = max(param_dict.values())
    best_param = list(param_dict.keys())[list(param_dict.values()).index(best_score)]

    # refit
    num_component = best_param
    pca = PCA(n_components=num_component)
    X_train_reduced = pca.fit_transform(X_train_val)
    X_test_reduced = pca.transform(X_test)
    # ols
    reg = LinearRegression().fit(X_train_reduced, Y_train_val)
    Y_pred = reg.predict(X_test_reduced)
    Y_pred_dict['PCA'] = Y_pred.item(0)
    best_param_dict['PCA'] = best_param

    # 6. PLS
    param_dict = {}
    for n in num_component_grid:
        pls = PLSRegression(n_components=n)
        param_dict[n] = pls.fit(X_train, Y_train).score(X_val, Y_val)

    best_score = max(param_dict.values())
    best_param = list(param_dict.keys())[list(param_dict.values()).index(best_score)]

    # refit
    pls = PLSRegression(n_components=best_param)
    reg = pls.fit(X_train_val, Y_train_val)
    Y_pred = reg.predict(X_test)
    Y_pred_dict['PLS'] = Y_pred.item(0)
    best_param_dict['PLS'] = best_param

    # 7. Gradient boost tree regression
    param_dict = {}
    for params, index in zip(ParameterGrid(param_grid_gbrt), range(len(list(ParameterGrid(param_grid_gbrt))))):
        param_dict[index] = GradientBoostingRegressor(learning_rate=params['learning_rate'],
                                                      n_estimators=params['n_estimators'],
                                                      min_samples_leaf=params['min_samples_leaf'],
                                                      max_depth=params['max_depth']).fit(X_train, Y_train.values.ravel()).score(X_val, Y_val)

    best_score = max(param_dict.values())
    best_param_index = list(param_dict.keys())[list(param_dict.values()).index(best_score)]
    best_param = list(ParameterGrid(param_grid_gbrt))[best_param_index]

    # refit
    reg = GradientBoostingRegressor(learning_rate=best_param['learning_rate'],
                                    n_estimators=best_param['n_estimators'],
                                    min_samples_leaf=best_param['min_samples_leaf'],
                                    max_depth=best_param['max_depth']).fit(X_train_val, Y_train_val.values.ravel())
    Y_pred = reg.predict(X_test)
    Y_pred_dict['GBRT'] = Y_pred.item(0)
    GBRT_param_dict = best_param

    # 8. Random Forrest regression
    param_dict = {}
    for params, index in zip(ParameterGrid(param_grid_rf), range(len(list(ParameterGrid(param_grid_rf))))):
        if params['max_features'] > len(X_train.columns.values):
            pass
        else:
            param_dict[index] = RandomForestRegressor(max_depth=params['max_depth'],
                                                      max_features=params['max_features'],
                                                      min_samples_leaf=params['min_samples_leaf'],
                                                      n_estimators=params['n_estimators']).fit(X_train, Y_train.values.ravel()).score(X_val, Y_val)

    best_score = max(param_dict.values())
    best_param_index = list(param_dict.keys())[list(param_dict.values()).index(best_score)]
    best_param = list(ParameterGrid(param_grid_rf))[best_param_index]

    # refit
    reg = RandomForestRegressor(max_depth=best_param['max_depth'],
                                max_features=best_param['max_features'],
                                min_samples_leaf=best_param['min_samples_leaf'],
                                n_estimators=best_param['n_estimators']).fit(X_train_val, Y_train_val.values.ravel())

    Y_pred = reg.predict(X_test)
    Y_pred_dict['RF'] = Y_pred.item(0)
    RF_param_dict = best_param

    # Convert the final result to dataframe
    result = pd.DataFrame.from_dict(Y_pred_dict, orient='index')
    result['date'] = pd.to_datetime(this_month) + relativedelta(day=31)
    result['target'] = target
    result = result.reset_index()
    result.columns = ['method', 'y_pred', 'date', 'target']

    # Store the best tuning parameters
    best_param_df = pd.DataFrame.from_dict(best_param_dict, orient='index').reset_index()
    RF_param_df = pd.DataFrame.from_dict(RF_param_dict, orient='index').reset_index()
    GBRT_param_df = pd.DataFrame.from_dict(GBRT_param_dict, orient='index').reset_index()

    RF_param_df['method'] = 'RF'
    GBRT_param_df['method'] = 'GBRT'
    RF_param_df = RF_param_df.rename(columns={'index': 'param', 0:'value'})
    GBRT_param_df = GBRT_param_df.rename(columns={'index': 'param', 0:'value'})

    best_param_df = best_param_df.rename(columns={'index': 'method', 0:'value'})
    best_param_df['param'] = np.nan

    best_param_df = pd.concat([best_param_df, RF_param_df, GBRT_param_df])
    best_param_df['date'] = pd.to_datetime(this_month) + relativedelta(day=31)
    best_param_df['target'] = target
    best_param_df = best_param_df.reset_index(drop=True)

    return result, best_param_df

