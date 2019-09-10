import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import sys
sys.path.insert(1, "/home/pvossler/ps_job")
# sys.path.append("/home/pvossler/ps_job")
import data
import parameters
from sklearn.covariance import MinCovDet, LedoitWolf
import utils
import datetime 



training_params = parameters.GetTrainingHyperParams(model)
# training_params['LAMBDA'] = 0.0078
# training_params['DELTA'] = 0.0078
training_params['LAMBDA'] = lambda_val
training_params['DELTA'] = delta_val
p = X_train.shape[1]

print(X_train.shape)
# chunk_list = [num_cuts] * (ncat)

    # Set the parameters for training deep knockoffs
pars = dict()

pars['avg_corr'] = avg_corr
pars['avg_corr_cat'] = avg_corr_cat
pars['avg_corr_cont'] = avg_corr_cont
# Number of epochs
pars['epochs'] = 25
# Number of iterations over the full data per epoch
pars['epoch_length'] = 100
# Data type, either "continuous" or "binary"
pars['family'] = "continuous"
# Dimensions of the data
pars['p'] = p
# List of categorical variables
pars['cat_var_idx'] = np.arange(0, (np.sum(num_cuts)))
# Number of discrete variables
pars['ncat'] = ncat
# Number of categories
pars['num_cuts'] = num_cuts
# Number of categories for each categorical variable
pars['chunk_list'] = num_cuts
# Size of regularizer
# pars['regularizer'] = grid_results[0]
# Boolean for using different weighting structure for decorr
pars['use_weighting'] = False
# Multiplier for weighting discrete variables
pars['kappa'] = 1
# Boolean for using the different decorr loss function from the paper
pars['diff_decorr'] = False
# Boolean for using mixed data in forward function
pars['mixed_data'] = True
# Size of the test set
pars['test_size'] = 0
# Batch size
pars['batch_size'] = int(0.5*n)
# Learning rate
pars['lr'] = 0.01
# When to decrease learning rate (unused when equal to number of epochs)
pars['lr_milestones'] = [pars['epochs']]
# Width of the network (number of layers is fixed to 6)
pars['dim_h'] = int(10*p)
# Penalty for the MMD distance
pars['GAMMA'] = training_params['GAMMA']
# Penalty encouraging second-order knockoffs
pars['LAMBDA'] = training_params['LAMBDA']
# Decorrelation penalty hyperparameter
pars['DELTA'] = training_params['DELTA']
# Target pairwise correlations between variables and knockoffs
pars['target_corr'] = corr_g
# Kernel widths for the MMD measure (uniform weights)
pars['alphas'] = [1., 2., 4., 8., 16., 32., 64., 128.]

pars_name = MODEL_DIRECTORY + 'pars' + '_p_' + str(p_size) + timestamp + '.npy'
# Save parameters
np.save(pars_name, pars)

# Where to store the machine
checkpoint_name = MODEL_DIRECTORY + model + timestamp + '_p_' + str(p_size) 

# Where to print progress information
logs_name = MODEL_DIRECTORY + model + timestamp + '_p_' + str(p_size)  + "_progress.txt"

# Initialize the machine
machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)

# Train the machine
machine.train(X_train)

print(timestamp)