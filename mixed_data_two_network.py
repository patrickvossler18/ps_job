import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
# from DeepKnockoffs import GaussianKnockoffs
import gk
import data
import parameters

# Number of features
p = 50

# Load the built-in multivariate Student's-t model and its default parameters
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
model = "mixed_student"
# model = "mixed"
distribution_params = parameters.GetDistributionParams(model, p)

# Initialize the data generator
DataSampler = data.DataSampler(distribution_params)

# Number of training examples
n = 10000
ncat = 10
cat_columns = np.arange(0, ncat)
num_cuts = 4
regularizer = 1e-4

# USE THIS FOR JUST K DUMMY VARIABLES
X_train = pd.DataFrame(DataSampler.sample(n))
X_train.iloc[:, cat_columns] = X_train.iloc[:, cat_columns].apply(lambda x: pd.qcut(x, 4, retbins=False, labels=False), axis=0).astype(str)
X_train_dums = pd.get_dummies(X_train.iloc[:, cat_columns], prefix=X_train.iloc[:, cat_columns].columns.values.astype(str).tolist())
X_train_cont = X_train.drop(cat_columns, axis=1)

SigmaHat_dis = np.cov(X_train_dums, rowvar=False)
SigmaHat_cont = np.cov(X_train_cont, rowvar=False)


# Initialize generator of second-order knockoffs
second_order_dis = gk.GaussianKnockoffs(SigmaHat_dis, mu=np.mean(X_train_dums, 0), method="sdp", regularizer=regularizer)
corr_g_dis = (np.diag(SigmaHat_dis) - np.diag(second_order_dis.Ds)) / np.diag(SigmaHat_dis)

second_order_cont = gk.GaussianKnockoffs(SigmaHat_cont, mu=np.mean(X_train_cont, 0), method="sdp", regularizer=regularizer)
corr_g_cont = (np.diag(SigmaHat_cont) - np.diag(second_order_cont.Ds)) / np.diag(SigmaHat_cont)


training_params = parameters.GetTrainingHyperParams(model)
p_dis = X_train_dums.shape[1]
p_cont = X_train_cont.shape[1]

def train_network(X, p,corr_g, data_type):
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
    # Multiplier for weighting discrete variables
    pars['kappa'] = 50
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
    np.save('/artifacts/pars_'+data_type+'.npy', pars)
    # Where to store the machine
    checkpoint_name = "/artifacts/" + model +"_"+data_type
    # Where to print progress information
    logs_name = "/artifacts/" + model + "_progress.txt"
    # Initialize the machine
    machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)
    # Train the machine
    machine.train(X.values)

train_network(X_train_dums,p_dis,corr_g_dis,"discrete")
train_network(X_train_cont,p_cont,corr_g_cont,"continuous")
