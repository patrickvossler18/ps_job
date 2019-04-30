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
black_data = pd.read_csv("black_clean.csv")

factor_list = pd.read_csv("black_factor_list.csv").values.tolist()
chunk_list = pd.read_csv("black_chunk_list.csv").values.tolist()
cat_var_idx = pd.read_csv("black_cat_var_idx.csv").values.tolist()
factor_list = list(chain(*factor_list))
chunk_list = list(chain(*chunk_list))
cat_var_idx = list(chain(*cat_var_idx))
cat_var_idx = np.array(cat_var_idx) - 1



# Drop Y and W
X =  black_data.drop(columns=["Y","W"])


# Split train test 80-20 and save index of train data for later
X_train = X.sample(frac=0.8,random_state=200)
np.savetxt("/artifacts/black_train_msk.csv", X_train.index, delimiter=",")

# Regularize the covariance and generate second order knockoffs

# SigmaHat_lw = ledoit_wolf(X_train)[0]
mcd = MinCovDet().fit(X_train)
SigmaHat_mcd = mcd.covariance_ 
# # np.min(np.linalg.eigvals(SigmaHat_mcd))
# # SigmaHat_mcd = SigmaHat_mcd + (1e-6)*np.eye(SigmaHat_mcd.shape[0])
# # np.min(np.linalg.eigvals(SigmaHat_mcd))
if(np.min(np.linalg.eigvals(SigmaHat_mcd)) < 0):
    SigmaHat_mcd = SigmaHat_mcd + (9e-3)*np.eye(SigmaHat_mcd.shape[0])

# X_train = X_train.values
# myScaler = preprocessing.StandardScaler()
# X_train = myScaler.fit_transform(X_train)
# emp_cov = covariance.empirical_covariance(X_train)
# shrunk_cov = covariance.shrunk_covariance(np.cov(X_train, rowvar=False), shrinkage=0.3)


SigmaHat = SigmaHat_mcd
# SigmaHat = np.cov(X_train, rowvar=False)
# if(np.min(np.linalg.eigvals(SigmaHat)) < 0):
#     SigmaHat = SigmaHat + (2e-3)*np.eye(SigmaHat.shape[0])

# second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=0.001)
second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=0.001)
# corr_g = np.nan_to_num((np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat))
corr_g = ((np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat))

print(np.average(corr_g))

training_params = parameters.GetTrainingHyperParams(model)
p = X_train.shape[1]
n = X_train.shape[0]

a = np.linspace(0.01,0.001,num=5)
b = np.linspace(0.01,0.001,num=5)

param_combos = [(x,y) for x in a for y in b]

# a = [0.0010,0.0055,0.0100]
# b = [0.0100,0.0078,0.0078]
# param_combos = list(zip(a,b))


for combo in param_combos:
    model="mstudent"
    print(combo)
    # Set the parameters for training deep knockoffs
    pars = dict()
    # Number of epochs
    pars['epochs'] = 25
    # Number of iterations over the full data per epoch
    pars['epoch_length'] = 50
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
    pars['batch_size'] = int(0.3*n)
    # Learning rate
    pars['lr'] = 0.01
    # When to decrease learning rate (unused when equal to number of epochs)
    pars['lr_milestones'] = [pars['epochs']]
    # Width of the network (number of layers is fixed to 6)
    pars['dim_h'] = int(10*p)
    # Penalty for the MMD distance
    pars['GAMMA'] = training_params['GAMMA']
    # Penalty encouraging second-order knockoffs
    pars['LAMBDA'] = combo[0]
    # Decorrelation penalty hyperparameter
    pars['DELTA'] = combo[1]
    # Target pairwise correlations between variables and knockoffs
    pars['target_corr'] = corr_g
    # Kernel widths for the MMD measure (uniform weights)
    pars['alphas'] = [1., 2., 4., 8., 16., 32., 64., 128.]

    par_lambda = str(np.round(combo[0],4))
    par_delta = str(np.round(combo[1],4))
    # Save parameters
    np.save('/artifacts/pars' + "_" + par_lambda + "_" + par_delta+'.npy', pars)

    model = model + "_" + par_lambda + "_" + par_delta

    # Where to store the machine
    checkpoint_name = "/artifacts/" + model

    # Where to print progress information
    logs_name = "/artifacts/" + model + "_progress.txt"

    # Initialize the machine
    machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)

    # Train the machine
    machine.train(X_train.values)
