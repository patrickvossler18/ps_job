import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import gk
import data
import parameters
from sklearn.covariance import MinCovDet

# Number of features
p = 100

# Load the built-in multivariate Student's-t model and its default parameters
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
# model = "mixed_student"
model = "mixed_student"
distribution_params = parameters.GetDistributionParams(model, p)

# Initialize the data generator
DataSampler = data.DataSampler(distribution_params)

# Number of training examples
n = 1000
ncat = int(p/2)
cat_columns = np.arange(0, ncat)
num_cuts = 4


# USE THIS FOR JUST K DUMMY VARIABLES
X_train = pd.DataFrame(DataSampler.sample(n))
X_train.iloc[:, cat_columns] = X_train.iloc[:, cat_columns].apply(lambda x: pd.qcut(x, 4, retbins=False, labels=False), axis=0).astype(str)
X_train_dums = pd.get_dummies(X_train.iloc[:, cat_columns], prefix=X_train.iloc[:, cat_columns].columns.values.astype(str).tolist())
X_train = pd.concat([X_train_dums.reset_index(drop=True), X_train.drop(cat_columns, axis=1).reset_index(drop=True)], axis=1)


SigmaHat = np.cov(X_train, rowvar=False)
 

# regularizer = np.array([1e-4]*(num_cuts*ncat)+[1e-4]*(SigmaHat.shape[1]-(num_cuts*ncat)))
# Initialize generator of second-order knockoffs

second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=1e-1)

# Measure pairwise second-order knockoff correlations
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

print(np.average(corr_g))
print(np.average(corr_g[1:(num_cuts*ncat)]))
print(np.average(corr_g[((num_cuts*ncat)+1):((num_cuts*ncat)+ int(p/2))]))

training_params = parameters.GetTrainingHyperParams(model)
p = X_train.shape[1]

chunk_list = [num_cuts] * (ncat)

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
pars['ncat'] = ncat
# List of categorical variables
pars['cat_var_idx'] = np.arange(0, (ncat * (num_cuts)))
# Number of discrete variables
pars['ncat'] = ncat
# Number of categories
pars['num_cuts'] = num_cuts
# Number of categories for each categorical variable
pars['chunk_list'] = chunk_list
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
