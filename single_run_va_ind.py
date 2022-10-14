import numpy as np
import pandas as pd
import datetime

import torch
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import PredefinedSplit,GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import *
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import statsmodels.api as sm
import statsmodels.formula.api as smf
from tuning import *
from my_nn import *
from scipy.stats.mstats import winsorize
pd.options.mode.chained_assignment = None  # default='warn'
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
import warnings
simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def get_data(this_month, data, train_span, valid_span, test_span, predictor_list, target, win=False):
    '''
    :param this_month: Prediction start date, should be str and the first day of month like '2022-01-01'
    :param data: Dataframe that contains dependent and independent variables
    :param train_span: Number of months used to train the model
    :param valid_span: Number of months used to validate the model
    :param test_span: Numbers of months that dependent variable lagged
    :param predictor_list: Variables that are used to predict
    :param target: Dependent variable
    :param win: If True then winsorize the dependent variable
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
    if win is True:
        # winsorize the target at 5% and 95% level. in this version, we winsorize the train and train_val sample separately.
        data_train_val = data[(data['date'] >= train_start) & (data['date'] < test_start)]
        data_train = data_train_val[(data_train_val['date'] >= train_start) & (data_train_val['date'] < valid_start)]
        data_valid = data_train_val[(data_train_val['date'] >= valid_start) & (data_train_val['date'] < test_start)]
        data_test = data[(data['date'] >= test_start) & (data['date'] < test_end)]
        data_valid['%s' % target] = winsorize(data_valid['%s' % target], limits=[0.05, 0.05])
        data_train_val['%s' % target] = winsorize(data_train_val['%s' % target], limits=[0.05, 0.05])
    else:
        data_train = data[(data['date'] >= train_start) & (data['date'] < valid_start)]
        data_valid = data[(data['date'] >= valid_start) & (data['date'] < test_start)]
        data_test = data[(data['date'] >= test_start) & (data['date'] < test_end)]

    # split X and Y
    Y_train = data_train[['%s' % target]]
    X_train = data_train[predictor_list]
    Y_val = data_valid[['%s' % target]]
    X_val = data_valid[predictor_list]
    Y_test = data_test[['%s' % target, 'permno']]
    X_test = data_test[predictor_list]
    # take the winsorized training and validation sample
    if win is True:
        Y_train_val = data_train_val[['%s' % target]]
        X_train_val = data_train_val[predictor_list]
    else:
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


def model_train(this_month, data, train_span, valid_span, test_span, predictor_list, target, win=False):
    '''

    :param this_month: Prediction start date, should be str and the first day of month like '2022-01-01'
    :param data: Dataframe that contains dependent and independent variables
    :param train_span: Number of months used to train the model
    :param valid_span: Number of months used to validate the model
    :param test_span: Number of months will be predicted, normally it needs to be one to work in this framework
    :param predictor_list: Variables that are used to predict
    :param target: Dependent variable
    :param win: If True then winsorize the dependent variable
    :return: Dataframe with four columns: predicted method, predicted value, date and name of dependent variable
    '''
    print("=" * 20, this_month, 'training', "=" * 20)

    # get data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_train_val, X_train_val = get_data(this_month, data, train_span, valid_span, test_span, predictor_list, target, win)
    # allocate space
    # Y_pred_dict = {}
    best_param_dict = {}
    model_dict = {}

    ####################################################################
    # get result from all other methods

    # 1. individual regression
    for j in predictor_list:
        reg = LinearRegression().fit(X_train_val[[j]], Y_train_val)
        model_dict['%s' % j] = reg
        # Y_pred = reg.predict(X_test[[j]])
        # Y_pred_dict['%s' % j] = Y_pred.item(0)

    #### combination method
    # df = pd.DataFrame.from_dict(Y_pred_dict, orient='index')
    # df.index = X_train_val.columns.values
    # print(df)

    # mean combinations
    # Y_pred = df.mean()
    # Y_pred_dict['Mean Combination'] = Y_pred.values.item(0)

    # median combinations
    # Y_pred = df.median()
    # Y_pred_dict['Median Combination'] = Y_pred.values.item(0)

    # 2. ols all
    reg = LinearRegression().fit(X_train_val, Y_train_val)
    model_dict['ols'] = reg
    # Y_pred = reg.predict(X_test)
    # Y_pred_dict['OLS'] = Y_pred.item(0)

    # 3. Lasso
    param_dict = {}
    for alpha in param_alpha:
        param_dict[alpha] = Lasso(alpha=alpha).fit(X_train, Y_train.values.ravel()).score(X_val, Y_val)

    best_score = max(param_dict.values())
    best_param = list(param_dict.keys())[list(param_dict.values()).index(best_score)]
    reg = Lasso(alpha=best_param).fit(X_train_val, Y_train_val.values.ravel())
    # refit
    # Y_pred = reg.predict(X_test)
    # Y_pred_dict['Lasso'] = Y_pred.item(0)
    best_param_dict['Lasso'] = best_param
    model_dict['Lasso'] = reg

    # 4. Ridge
    param_dict = {}
    for alpha in param_alpha:
        param_dict[alpha] = Ridge(alpha=alpha).fit(X_train, Y_train.values.ravel()).score(X_val, Y_val)

    best_score = max(param_dict.values())
    best_param = list(param_dict.keys())[list(param_dict.values()).index(best_score)]
    reg = Ridge(alpha=best_param).fit(X_train_val, Y_train_val.values.ravel())

    # Y_pred = reg.predict(X_test)
    # Y_pred_dict['Ridge'] = Y_pred.item(0)
    best_param_dict['Ridge'] = best_param
    model_dict['Ridge'] = reg

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
    # Y_pred = reg.predict(X_test_reduced)
    # Y_pred_dict['PCA'] = Y_pred.item(0)
    best_param_dict['PCA'] = best_param
    model_dict['PCA'] = pca
    model_dict['PCA_reg'] = reg

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
    # Y_pred = reg.predict(X_test)
    # Y_pred_dict['PLS'] = Y_pred.item(0)
    best_param_dict['PLS'] = best_param
    model_dict['PLS'] = reg

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
    # Y_pred = reg.predict(X_test)
    # Y_pred_dict['GBRT'] = Y_pred.item(0)
    GBRT_param_dict = best_param
    model_dict['GBRT'] = reg

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

    # Y_pred = reg.predict(X_test)
    # Y_pred_dict['RF'] = Y_pred.item(0)
    RF_param_dict = best_param
    model_dict['RF'] = reg

    # 9. Neural Network (2 layer)
    n_feature = len(X_train.columns)
    # transform data to tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
    X_train_val_tensor = torch.tensor(X_train_val.values, dtype=torch.float32)
    Y_train_val_tensor = torch.tensor(Y_train_val.values, dtype=torch.float32)

    # find the best parameters
    param_dict = {}
    for params, index in zip(ParameterGrid(param_grid_nn), range(len(list(ParameterGrid(param_grid_nn))))):
        net2 = Net2(n_feature=n_feature, n_hidden1=16, n_hidden2=4, n_output=1).to(device)  # define the network

        optimizer = torch.optim.SGD(net2.parameters(), lr=params['NN_learning_rate'], weight_decay=params['NN_alpha'])
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

        for epoch in range(params['NN_max_iter']):
            prediction = net2(X_train_tensor)  # input x and predict based on x

            loss = loss_func(prediction, Y_train_tensor)  # must be (1. nn output, 2. target)

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        # calculate in-sample R2
        y_pred_temp = net2(X_train_tensor)
        param_dict[index] = 1 - (((Y_train_tensor - y_pred_temp) ** 2).sum() / ((Y_train_tensor - y_pred_temp.mean()) ** 2).sum()).item()

    best_score = max(param_dict.values())
    best_param_index = list(param_dict.keys())[list(param_dict.values()).index(best_score)]
    best_param = list(ParameterGrid(param_grid_nn))[best_param_index]

    # refit
    net2 = Net2(n_feature=n_feature, n_hidden1=16, n_hidden2=4, n_output=1).to(device)  # define the network

    optimizer = torch.optim.SGD(net2.parameters(), lr=best_param['NN_learning_rate'], weight_decay=best_param['NN_alpha'])
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    for epoch in range(best_param['NN_max_iter']):
        prediction = net2(X_train_val_tensor)  # input x and predict based on x

        loss = loss_func(prediction, Y_train_val_tensor)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    NN2_param_dict = best_param
    model_dict['NN2'] = net2

    # 10. Neural Network (4 layer)
    n_feature = len(X_train.columns)
    # transform data to tensor
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
    X_train_val_tensor = torch.tensor(X_train_val.values, dtype=torch.float32)
    Y_train_val_tensor = torch.tensor(Y_train_val.values, dtype=torch.float32)

    # find the best parameters
    param_dict = {}
    for params, index in zip(ParameterGrid(param_grid_nn), range(len(list(ParameterGrid(param_grid_nn))))):
        net4 = Net4(n_feature=n_feature, n_hidden1=32, n_hidden2=16, n_hidden3=8, n_hidden4=4, n_output=1).to(device)  # define the network

        optimizer = torch.optim.SGD(net4.parameters(), lr=params['NN_learning_rate'], weight_decay=params['NN_alpha'])
        loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

        for epoch in range(params['NN_max_iter']):
            prediction = net4(X_train_tensor)  # input x and predict based on x

            loss = loss_func(prediction, Y_train_tensor)  # must be (1. nn output, 2. target)

            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

        # calculate in-sample R2
        y_pred_temp = net4(X_train_tensor)
        param_dict[index] = 1 - (((Y_train_tensor - y_pred_temp) ** 2).sum() / ((Y_train_tensor - y_pred_temp.mean()) ** 2).sum()).item()

    best_score = max(param_dict.values())
    best_param_index = list(param_dict.keys())[list(param_dict.values()).index(best_score)]
    best_param = list(ParameterGrid(param_grid_nn))[best_param_index]

    # refit
    net4 = Net4(n_feature=n_feature, n_hidden1=32, n_hidden2=16, n_hidden3=8, n_hidden4=4, n_output=1).to(device)  # define the network

    optimizer = torch.optim.SGD(net4.parameters(), lr=best_param['NN_learning_rate'], weight_decay=best_param['NN_alpha'])
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    for epoch in range(best_param['NN_max_iter']):
        prediction = net4(X_train_val_tensor)  # input x and predict based on x

        loss = loss_func(prediction, Y_train_val_tensor)  # must be (1. nn output, 2. target)

        optimizer.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

    NN4_param_dict = best_param
    model_dict['NN4'] = net4

    # Convert the final result to dataframe
    # result = pd.DataFrame.from_dict(Y_pred_dict, orient='index')
    # result['date'] = pd.to_datetime(this_month) + relativedelta(day=31)
    # result['target'] = target
    # result = result.reset_index()
    # result.columns = ['method', 'y_pred', 'date', 'target']
    # if len(Y_test['%s' % target].values) == 1:  # cannot add real return of individual firms
    #     result['Y'] = float(Y_test['%s' % target].values)
    # else:
    #     pass

    # Store the best tuning parameters
    best_param_df = pd.DataFrame.from_dict(best_param_dict, orient='index').reset_index()
    RF_param_df = pd.DataFrame.from_dict(RF_param_dict, orient='index').reset_index()
    GBRT_param_df = pd.DataFrame.from_dict(GBRT_param_dict, orient='index').reset_index()
    NN2_param_df = pd.DataFrame.from_dict(NN2_param_dict, orient='index').reset_index()
    NN4_param_df = pd.DataFrame.from_dict(NN4_param_dict, orient='index').reset_index()

    RF_param_df['method'] = 'RF'
    GBRT_param_df['method'] = 'GBRT'
    NN2_param_df['method'] = 'NN2'
    NN4_param_df['method'] = 'NN4'
    RF_param_df = RF_param_df.rename(columns={'index': 'param', 0:'value'})
    GBRT_param_df = GBRT_param_df.rename(columns={'index': 'param', 0:'value'})
    NN2_param_df = NN2_param_df.rename(columns={'index': 'param', 0: 'value'})
    NN4_param_df = NN4_param_df.rename(columns={'index': 'param', 0: 'value'})

    best_param_df = best_param_df.rename(columns={'index': 'method', 0:'value'})
    best_param_df['param'] = np.nan

    best_param_df = pd.concat([best_param_df, RF_param_df, GBRT_param_df, NN2_param_df, NN4_param_df])
    best_param_df['date'] = pd.to_datetime(this_month) + relativedelta(day=31)
    best_param_df['target'] = target
    best_param_df = best_param_df.reset_index(drop=True)

    return model_dict, best_param_df


def single_run(model_dict, this_month, data, train_span, valid_span, test_span, predictor_list, target, win=False):
    '''

    :param this_month: Prediction start date, should be str and the first day of month like '2022-01-01'
    :param data: Dataframe that contains dependent and independent variables
    :param train_span: Number of months used to train the model
    :param valid_span: Number of months used to validate the model
    :param test_span: Number of months will be predicted, normally it needs to be one to work in this framework
    :param predictor_list: Variables that are used to predict
    :param target: Dependent variable
    :param win: If True then winsorize the dependent variable
    :return: Dataframe with four columns: predicted method, predicted value, date and name of dependent variable
    '''
    print("=" * 20, this_month, 'predicting', "=" * 20)

    # get data
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_train_val, X_train_val = get_data(this_month, data, train_span,
                                                                                        valid_span, test_span,
                                                                                        predictor_list, target, win)
    # reset X_test, Y_test index
    X_test = X_test.reset_index(drop=True)
    Y_test = Y_test.reset_index(drop=True)

    # allocate space
    Y_pred_df = pd.DataFrame()

    ####################################################################
    # get result from all other methods

    # 1. individual regression
    for j in predictor_list:
        reg = model_dict['%s' % j]
        Y_pred = reg.predict(X_test[[j]])
        Y_pred_df['%s' % j] = Y_pred.ravel()

    #### combination method

    # mean combinations
    Y_pred_df['Mean Combination'] = Y_pred_df.mean(axis=1)

    # median combinations
    Y_pred_df['Median Combination'] = Y_pred_df.drop('Mean Combination', axis=1).median(axis=1)

    # 2. ols all
    reg = model_dict['ols']
    Y_pred = reg.predict(X_test)
    Y_pred_df['OLS'] = Y_pred

    # 3. Lasso
    reg = model_dict['Lasso']
    # refit
    Y_pred = reg.predict(X_test)
    Y_pred_df['Lasso'] = Y_pred

    # 4. Ridge
    reg = model_dict['Ridge']
    # refit
    Y_pred = reg.predict(X_test)
    Y_pred_df['Ridge'] = Y_pred

    # 5. PCA
    # refit
    X_test_reduced = model_dict['PCA'].transform(X_test)
    # ols
    reg = model_dict['PCA_reg']
    Y_pred = reg.predict(X_test_reduced)
    Y_pred_df['PCA'] = Y_pred

    # 6. PLS
    # refit
    reg = model_dict['PLS']
    Y_pred = reg.predict(X_test)
    Y_pred_df['PLS'] = Y_pred

    # 7. Gradient boost tree regression
    reg = model_dict['GBRT']
    Y_pred = reg.predict(X_test)
    Y_pred_df['GBRT'] = Y_pred

    # 8. Random Forrest regression
    reg = model_dict['RF']
    Y_pred = reg.predict(X_test)
    Y_pred_df['RF'] = Y_pred

    # 9. Neural Network (2 layers)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    net = model_dict['NN2']
    Y_pred = net(X_test_tensor)
    Y_pred_df['NN2'] = pd.DataFrame(Y_pred.detach().numpy())[0]

    # 10. Neural Network (4 layers)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    net = model_dict['NN4']
    Y_pred = net(X_test_tensor)
    Y_pred_df['NN4'] = pd.DataFrame(Y_pred.detach().numpy())[0]

    # Convert the final result to dataframe
    Y_pred_df[['permno', 'y']] = Y_test[['permno', '%s' % target]].reset_index(drop=True)
    result = Y_pred_df.copy()
    result['date'] = pd.to_datetime(this_month) + relativedelta(day=31)
    result['target'] = target

    return result

