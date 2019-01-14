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
model = "gaussian"
distribution_params = parameters.GetDistributionParams(model, p)

# Initialize the data generator
DataSampler = data.DataSampler(distribution_params)

# Number of training examples
n = 10000
ncat = 10
cat_columns = np.arange(0,ncat)
num_cuts = 3

# Sample training data
# X_train = pd.DataFrame(DataSampler.sample(n))
# X_train.iloc[:,cat_columns] = X_train.iloc[:,cat_columns].apply(lambda x: pd.qcut(x, 4, retbins=False,labels=False), axis=0)
# obs_logits = X_train.iloc[:,cat_columns].apply(lambda x: x.value_counts(normalize=True), axis=0)

# logits_list = [obs_logits for i in range(0,n)]

# for j in range(len(logits_list)):
#     if j % 1000 == 0:
#         print(j)
#     a = logits_list[j].apply(lambda x: gumbel_softmax_sample(torch.FloatTensor(x.apply(np.log)),0.8)).unstack()
#     res = pd.concat([pd.DataFrame(a[i]).T.reset_index().drop(['index'],axis=1) for i in cat_columns ], axis = 1)
#     logits_list[j] = res

# logits_df = pd.concat(logits_list)
# X_train_cont = X_train.drop(cat_columns,axis=1)
# X_train = pd.concat([logits_df.reset_index(drop=True),X_train_cont.reset_index(drop=True) ], axis=1)
# res = pd.concat([pd.DataFrame(a[i]).T.reset_index().drop(['index'],axis=1) for i in range(0,num_cuts) ], axis = 0)
# Variable(torch.FloatTensor([[math.log(0.1), math.log(0.4), math.log(0.3), math.log(0.2)]] * 20000))
# Make the columns into categorical variables

## USE THIS FOR THE K-1 DUMMY VARIABLE TEST
# X_train = pd.DataFrame(DataSampler.sample(n))
# X_train.iloc[:,cat_columns] = X_train.iloc[:,cat_columns].apply(lambda x: pd.qcut(x, 4, retbins=False,labels=False), axis=0).astype(str)
# X_train_dums = pd.get_dummies(X_train.iloc[:,cat_columns],drop_first=True, prefix= X_train.iloc[:,cat_columns].columns.values.astype(str).tolist())
# X_train = pd.concat([X_train_dums.reset_index(drop=True),X_train.drop(cat_columns,axis = 1).reset_index(drop=True)], axis=1)

## USE THIS FOR JUST K DUMMY VARIABLES
X_train = pd.DataFrame(DataSampler.sample(n))
X_train.iloc[:,cat_columns] = X_train.iloc[:,cat_columns].apply(lambda x: pd.qcut(x, num_cuts, retbins=False,labels=False), axis=0).astype(str)
X_train_dums = pd.get_dummies(X_train.iloc[:,cat_columns], prefix= X_train.iloc[:,cat_columns].columns.values.astype(str).tolist())
X_train = pd.concat([X_train_dums.reset_index(drop=True),X_train.drop(cat_columns,axis = 1).reset_index(drop=True)], axis=1)



SigmaHat = np.cov(X_train, rowvar=False)

# Initialize generator of second-order knockoffs
second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train,0), method="sdp")
# X_tilde = second_order.generate(X_train)

# X_tilde.iloc[:,cat_columns] = X_tilde.iloc[:,cat_columns].apply(lambda x: pd.qcut(x, 4, retbins=False,labels=False), axis=0)


# Measure pairwise second-order knockoff correlations 
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)


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
# List of categorical variables
pars['cat_var_idx'] = np.arange(0,(ncat * (num_cuts)))
# Size of the test set
pars['test_size']  = 0
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
pars['alphas'] = [1.,2.,4.,8.,16.,32.,64.,128.]


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