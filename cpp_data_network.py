import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
# from DeepKnockoffs import GaussianKnockoffs
import gk
import data
import parameters
from sklearn.covariance import MinCovDet, LedoitWolf
from itertools import chain


# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
# model = "mixed_student"
model = "gaussian"

# Load data
cpp_data = pd.read_csv("cpp_final.csv")
factor_list = pd.read_csv("factor_list.csv").values.tolist()

factor_list = list(chain(*factor_list))

# Drop Y and W
X =  cpp_data.drop(columns=["Y","W"])
X_new = X

# Convert factors to dummies
chunk_list = []
for factor in factor_list:
    # expand the variable
    expanded = pd.get_dummies(data=X[factor])
    # count how many columns
    chunks = expanded.shape[1]
    chunk_list.append(chunks)
    X_new = pd.concat([X_new, expanded], axis=1)


X_new = X_new.drop(columns = factor_list)

cat_start = (X_new.shape[1]) - np.sum(chunk_list)
cat_var_idx = np.arange(cat_start,X_new.shape[1])

# Split train test 80-20 and save index of train data for later
X_train = X_new.sample(frac=0.8,random_state=200)
np.savetxt("/artifacts/train_msk.csv", X_train.index, delimiter=",")

# Regularize the covariance and generate second order knockoffs
# mcd = LedoitWolf().fit(X_train)
# SigmaHat_mcd = mcd.covariance_ 
SigmaHat_mcd = np.cov(X_train, rowvar=False)
second_order = gk.GaussianKnockoffs(SigmaHat_mcd, mu=np.mean(X_train, 0), method="sdp", regularizer=1e-3)
corr_g = (np.diag(SigmaHat_mcd) - np.diag(second_order.Ds)) / np.diag(SigmaHat_mcd)

print(np.average(corr_g))

training_params = parameters.GetTrainingHyperParams(model)
p = X_train.shape[1]
n = X_train.shape[0]

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
machine.train(X_train.values)
