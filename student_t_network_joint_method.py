import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
sys.path.insert(1, "/home/pvossler/ps_job")
# sys.path.append("/home/pvossler/ps_job")
import data
import parameters
from sklearn.covariance import MinCovDet, LedoitWolf
import utils
import datetime 

now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

MODEL_DIRECTORY = "/home/pvossler/cm_idea/"

MODEL_DIRECTORY = "/home/pvossler/cm_idea/"
print(p)
p_size = int(p)
p = int(p)

# Load the built-in multivariate Student's-t model and its default parameters
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
model = "mstudent"
distribution_params = parameters.GetDistributionParams(model, p)

# Initialize the data generator
DataSampler = data.DataSampler(distribution_params)

# Number of training examples
n = 1000

# not used but included in dictionary
ncat = int(p/2)
cat_columns = np.arange(0, ncat)
num_cuts = 4

# Sample training data
X_train = DataSampler.sample(n)

if robust_cov:
    mcd = MinCovDet().fit(X_train)
    SigmaHat = mcd.covariance_ 
else:
    SigmaHat = np.cov(X_train, rowvar=False)

SigmaHat= SigmaHat + (reg_val)*np.eye(SigmaHat.shape[0])

# TO USE LATER
# regularizer = np.array([1e-1]*(num_cuts*ncat)+[1e-1]*(SigmaHat.shape[1]-(num_cuts*ncat)))
# # Initialize generator of second-order knockoffs
# second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=1e-1)

# Initialize generator of second-order knockoffs
second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp")

# Measure pairwise second-order knockoff correlations
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

print(np.average(corr_g))

training_params = parameters.GetTrainingHyperParams(model)
training_params['LAMBDA'] = 0.001
training_params['DELTA'] = 0.05


p = X_train.shape[1]

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
# Size of regularizer
# pars['regularizer'] = grid_results[0]
# Boolean for using different weighting structure for decorr
pars['use_weighting'] = False
# Boolean for using mixed data in forward function
pars['mixed_data'] = False
# Multiplier for weighting discrete variables
pars['kappa'] = 1
# Boolean for using the different decorr loss function from the paper
pars['diff_decorr'] = False
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
