import numpy as np

# the purpose of tuning.py is to save tuning parameter for all methods
############################################
# parameter for cross_validation
# define the grid search parameters

# 1. for Lasso, Ridge & Elastic Net
param_alpha = np.exp(np.arange(-20, 10.5, step=0.5))
l1_ratio = [.01, .05, .1, .3, .5, .7, .9, .95, .99]  # 0.5

# 2. PCA and PLS
num_component_grid = [3, 4, 5, 6, 7]

# 3. for Gradient Boost
GBtree_learning_rate = [0.2, 0.1]
GBtree_max_depth = [4, 5]
GBtree_n_estimators = [50, 100]
GBtree_min_samples_leaf = [10]

param_grid_gbrt = {'learning_rate': GBtree_learning_rate,
                  'max_depth': GBtree_max_depth,
                  'n_estimators': GBtree_n_estimators,
                   'min_samples_leaf': GBtree_min_samples_leaf}

# aggregate market
# 4. for Random Forest
RF_n_estimators = [500]
RF_max_depth = [4, 5]
RF_max_features = [10, 20]
RF_min_samples_leaf = [10]

param_grid_rf = {'n_estimators': RF_n_estimators,
                 'max_depth': RF_max_depth,
                 'max_features': RF_max_features,
                 'min_samples_leaf': RF_min_samples_leaf}

# # indivdiual stock market
# # 4. for Random Forest
# RF_n_estimators = [500]
# RF_max_depth = [5, 7]
# RF_max_features = [10, 20]
# RF_min_samples_leaf = [5]

# 5. Neural Network
NN_alpha = [0.01, 0.001]
NN_learning_rate = [0.01, 0.001]
NN_max_iter = [100,200]

param_grid_nn = {'NN_alpha': NN_alpha,
                 'NN_learning_rate': NN_learning_rate,
                 'NN_max_iter': NN_max_iter}
