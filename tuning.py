# fix random seed
# from numpy.random import seed
# seed(2019)
# import tensorflow
# tensorflow.random.set_seed(2019)
# from tensorflow import set_random_seed
# set_random_seed(2019)

# from keras import losses
# from keras import optimizers, initializers
import numpy as np

# the purpose of tuning.py is to save tuning parameter for all NN
# which means all NN use same tuning from here
############################################
# parameter for fitting the model NN

# learning_rate = 0.001
# #learning_rate = 0.0001 #nn2-combi
# initial_sd = 0.1
# optimizer = optimizers.RMSprop(lr=learning_rate)
# lossfun = losses.mean_squared_error
# initializer = initializers.random_normal(stddev=initial_sd)

# param_grid = {
#     "alpha_nn": np.exp(np.arange(-10, 10, step=1))
# }

#param_alpha_nn = 1e-4

# nn_list_alpha = np.exp(np.arange(-10, 5, step=2))
#
# nb_epoch = 300
# batch_size = 1000


############################################
# LASSO ENet
# param_grid = {
#     "alpha_nn": np.exp(np.arange(-20, 20, step=1))
# }

param_alpha = np.exp(np.arange(-40, 40, step=.5))
l1_ratio = [0, .1, .3, .5, .7, .9, 1]

# n_iter = 10000

############################################
# tree

############################################
# tree

# GBtree_learning_rate = [0.1, 0.05, 0.01]
# GBtree_max_depth = [3, 5, 10]
# GBtree_n_estimators = [100, 200, 500]
# GBtree_subsample = [0.5, 0.9, 1]

# RF_n_estimators = [200, 500, 1000]
# RF_max_depth = [4, 5, 6]
# RF_max_features = ['sqrt','log2']

RF_n_estimators = [1000]
RF_max_features = ['sqrt']
# RF_max_depth = [4, 5, 6]
RF_min_samples_leaf = [200, 400]


param_grid_rf = {'n_estimators': RF_n_estimators,
              # 'max_depth': RF_max_depth,
              'max_features': RF_max_features,
              'min_samples_leaf': RF_min_samples_leaf}

GBtree_learning_rate = [0.1, 0.05]
GBtree_min_samples_leaf = [200]
GBtree_max_depth = [3, 5]
GBtree_n_estimators = [100, 200]
#GBtree_max_features = ['sqrt']


param_grid_gbrt = {'learning_rate': GBtree_learning_rate,
               'max_depth': GBtree_max_depth,
              'n_estimators': GBtree_n_estimators,
              #'max_features': GBtree_max_features,
              'min_samples_leaf': GBtree_min_samples_leaf
              }

############################################
# mean median combination TRIM BAR
# trim_bar = 0.1

############################################
# subset selection

# n_max_subset = 3

############################################
# autoencoder / pca / pls

# ae_list_n_comp = [1,3,5]
# pca_list_n_comp = [1,2,3,4,5]
# pls_list_n_comp = [1,2,3,4,5]

param_grid_pipe_pca3 = {
    'pca__n_components': [3]
}

param_grid_pipe_pca5 = {
    'pca__n_components': [5]
}

param_grid_pipe_pls3 = {
    'pls__n_components': [3]
}

param_grid_pipe_pls5 = {
    'pls__n_components': [5]
}

# # backward selection
# param_grid_pipe_bst = {
#     'back__fK': [5]
# }
