import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
# from DeepKnockoffs import GaussianKnockoffs
import gk
import data
import parameters
from sklearn.covariance import MinCovDet, LedoitWolf, ledoit_wolf, GraphicalLassoCV,empirical_covariance
from sklearn import preprocessing, covariance
from itertools import chain


# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
# model = "mixed_student"
model = "mixed_student"

# Load data
black_data = pd.read_csv("black_clust.csv")
factor_list = pd.read_csv("black_factor_list.csv").values.tolist()
chunk_list = pd.read_csv("black_chunk_list.csv").values.tolist()
cat_var_idx = pd.read_csv("black_cat_var_idx.csv").values.tolist()
factor_list = list(chain(*factor_list))
chunk_list = list(chain(*chunk_list))
cat_var_idx = list(chain(*cat_var_idx))
cat_var_idx = np.array(cat_var_idx) - 1


# Drop Y and W
X = black_data.drop(columns=["Y", "W"])


# Split train test 80-20 and save index of train data for later
# X_train = X.sample(frac=0.8,random_state=200)
X_train = X.sample(frac=0.2, random_state=200)
np.savetxt("/artifacts/black_train_msk.csv", X_train.index, delimiter=",")

# Regularize the covariance and generate second order knockoffs
mcd = MinCovDet().fit(X_train)
SigmaHat_mcd = mcd.covariance_ 

SigmaHat_mcd = SigmaHat_mcd + (1e-3)*np.eye(SigmaHat_mcd.shape[0])


SigmaHat = SigmaHat_mcd

# second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=0.001)
second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=0)
# corr_g = np.nan_to_num((np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat))
corr_g = ((np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat))

print(np.average(corr_g))

training_params = parameters.GetTrainingHyperParams(model)
p = X_train.shape[1]
n = X_train.shape[0]


training_params['LAMBDA'] = 0.001
training_params['DELTA'] = 0.0078

# Set the parameters for training deep knockoffs
pars = dict()
# Number of epochs
pars['epochs'] = 50
# Number of iterations over the full data per epoch
pars['epoch_length'] = 100
# Data type, either "continuous" or "binary"
pars['family'] = "continuous"
# Dimensions of the data
pars['p'] = p
# List of categorical variables
pars['cat_var_idx'] = cat_var_idx
# Number of categories for each categorical variable
pars['chunk_list'] = chunk_list
# # Number of categories
# pars['num_cuts'] = num_cuts
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
pars['batch_size'] = int(0.2*n)
# pars['batch_size'] = 100
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

# machine = KnockoffMachine(pars)
# machine.train(X_train.values)

# Save parameters
np.save('/artifacts/pars.npy', pars)


# Where to store the machine
checkpoint_name = "/artifacts/" + model

# Where to print progress information
logs_name = "/artifacts/" + model + "_progress.txt"

# Initialize the machine
machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)

# Train the machine
machine.train(X_train)
