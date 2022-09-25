import numpy as np

# the purpose of tuning.py is to save tuning parameter for all methods
############################################
# parameter for cross_validation
# define the grid search parameters
# 1. for Lasso, Ridge & Elastic Net
param_alpha = np.exp(np.arange(-20, 10.5, step=0.5))
l1_ratio = [.01, .05, .1, .3, .5, .7, .9, .95, .99]  # 0.5

# 2. PCA and PLS
num_component = 3

# 3. for Gradient Boost
GBtree_learning_rate = [0.2, 0.1]
GBtree_max_depth = [5, 7]
GBtree_n_estimators = [50, 100]

# 4. for Random Forest
RF_n_estimators = [500]
RF_max_depth = [5, 7]
RF_max_features = [10, 20]
RF_min_samples_leaf = [5]

# # 2. for Gradient Boost
# GBtree_learning_rate = [0.2, 0.1]
# GBtree_max_depth = [3, 6]
# GBtree_n_estimators = [100]
#
# # 3. for Random Forest
# RF_n_estimators = [200]
# RF_max_depth = [3, 6]
# RF_max_features = [10, 20]