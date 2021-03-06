import numpy as np
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import data
import parameters
from sklearn.covariance import MinCovDet, LedoitWolf
from scipy.linalg import toeplitz, cholesky

import datetime 
now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

MODEL_DIRECTORY = "/home/pvossler/cm_idea/"

# Number of features
p = 50

# Load the built-in multivariate Student's-t model and its default parameters
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
model = "gaussian"
distribution_params = parameters.GetDistributionParams(model, p)

# Initialize the data generator
DataSampler = data.DataSampler(distribution_params)

# Number of training examples
n = 1000

# not used but included in dictionary
# ncat = p/2
# cat_columns = np.arange(0, ncat)
# num_cuts = 4

# Sample training data
X_train = DataSampler.sample(n)


# simulation covariance matrix (AR(1) process)
# r = 0.5
# real_cov = toeplitz(r ** np.arange(p))
# coloring_matrix = cholesky(real_cov)
# X_train = np.dot(np.random.normal(size=(n, p)), coloring_matrix.T)


SigmaHat = np.cov(X_train, rowvar=False)
# mcd = MinCovDet().fit(X_train)
# SigmaHat = mcd.covariance_ 
# SigmaHat= SigmaHat + (2e-3)*np.eye(SigmaHat.shape[0])

# TO USE LATER
# regularizer = np.array([1e-1]*(num_cuts*ncat)+[0]*(SigmaHat.shape[1]-(num_cuts*ncat)))
# # Initialize generator of second-order knockoffs
# second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=regularizer)

# Initialize generator of second-order knockoffs
second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp")

# Measure pairwise second-order knockoff correlations
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

print(np.average(corr_g))

training_params = parameters.GetTrainingHyperParams(model)

p = X_train.shape[1]

# Set the parameters for training deep knockoffs
pars = dict()
# Number of epochs
pars['epochs'] = 100
# Number of iterations over the full data per epoch
pars['epoch_length'] = 100
# Data type, either "continuous" or "binary"
pars['family'] = "continuous"
# Dimensions of the data
pars['p'] = p
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


# Save parameters
np.save(MODEL_DIRECTORY + 'pars' + timestamp + '.npy', pars)

# Where to store the machine
checkpoint_name = MODEL_DIRECTORY + model + timestamp

# Where to print progress information
logs_name = MODEL_DIRECTORY + model + timestamp + "_progress.txt"

# Initialize the machine
machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)

# Train the machine
machine.train(X_train)
