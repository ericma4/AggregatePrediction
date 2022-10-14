import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from single_run_va_ind import *

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class Net2(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net2, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)  # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden2, n_output)  # output layer
    #
    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = self.predict(x)  # linear output
        return x


class Net4(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_output):
        super(Net4, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)  # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)  # hidden layer
        self.hidden3 = torch.nn.Linear(n_hidden2, n_hidden3)  # hidden layer
        self.hidden4 = torch.nn.Linear(n_hidden3, n_hidden4)  # hidden layer
        self.predict = torch.nn.Linear(n_hidden4, n_output)  # output layer
    #
    def forward(self, x):
        x = F.relu(self.hidden1(x))  # activation function for hidden layer
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = F.relu(self.hidden4(x))
        x = self.predict(x)  # linear output
        return x


# for test (ignore this part)
if __name__ == '__main__':
    # build up data
    data = pd.read_feather('/home/jianxin/chars/running/chars_rank_imputed.feather')
    #
    predictor_list = []
    #
    for i in list(data.columns.values):
        if i.startswith('rank_'):
            predictor_list.append(i)
        else:
            pass
    #
    result = pd.DataFrame()
    #
    param = pd.DataFrame()
    #
    # construct the time to split the data
    for year in range(1938, 1942):
        this_month = str(year) + '-' + str(1) + '-01'
    #
    X_train, X_val, X_test, Y_train, Y_val, Y_test, Y_train_val, X_train_val = get_data(this_month=this_month,
                                                                                        data=data, train_span=72,
                                                                                        valid_span=48, test_span=1,
                                                                                        predictor_list=predictor_list,
                                                                                        target='ret', win=False)
    # transform data to tensor
    n_feature = len(X_train.columns)
    #
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    #
    net2 = Net2(n_feature=n_feature, n_hidden1=16, n_hidden2=4, n_output=1).to(device)  # define the network
    print(net2)  # net architecture
    #
    optimizer = torch.optim.SGD(net2.parameters(), lr=0.01, weight_decay=0.01)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss
    #
    for epoch in range(2):
        prediction = net2(X_train_tensor)     # input x and predict based on x
        #
        loss = loss_func(prediction, Y_train_tensor)     # must be (1. nn output, 2. target)
        #
        optimizer.zero_grad()   # clear gradients for next train
        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients
    #
    # Prediction
    y_pred = net2(X_test_tensor)
    # y_pred = pd.DataFrame(y_pred).astype(float)


