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




p = 50
now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

MODEL_DIRECTORY = "/home/pvossler/cm_idea/"
print(p)
p_size = p
# Load the built-in multivariate Student's-t model and its default parameters
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
# model = "mixed_student"
model = "mixed"
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
X_train.iloc[:, cat_columns] = X_train.iloc[:, cat_columns].apply(lambda x: pd.qcut(x, num_cuts, retbins=False, labels=False), axis=0).astype(str)
X_train_dums = pd.get_dummies(X_train.iloc[:, cat_columns], prefix=X_train.iloc[:, cat_columns].columns.values.astype(str).tolist())
X_train = pd.concat([X_train_dums.reset_index(drop=True), X_train.drop(cat_columns, axis=1).reset_index(drop=True)], axis=1)

# 
if robust_cov:
    mcd = MinCovDet().fit(X_train)
    SigmaHat = mcd.covariance_ 
else:
    SigmaHat = np.cov(X_train, rowvar=False)

# reg_val = 5e-3
SigmaHat= SigmaHat + (reg_val)*np.eye(SigmaHat.shape[0])

# second_order = gk.GaussianKnockoffs(SigmaHat_mcd, mu=np.mean(X_train, 0), method="sdp", regularizer=1e-1)
# second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=1e-1)

second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp")

# Measure pairwise second-order knockoff correlations
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

print(np.average(corr_g))

print(np.average(corr_g[1:(np.sum(num_cuts)-1)]))
print(np.average(corr_g[((np.sum(num_cuts)-1)+1):]))

avg_corr = np.average(corr_g)

avg_corr_cat = np.average(corr_g[1:(np.sum(num_cuts)-1)])
avg_corr_cont = np.average(corr_g[((np.sum(num_cuts)-1)+1):])
