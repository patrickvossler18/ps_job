import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import sys
sys.path.insert(1, "/home/pvossler/ps_job")
import data
import parameters
from sklearn.covariance import MinCovDet, LedoitWolf
from sklearn import preprocessing, covariance
from itertools import chain
import utils
import datetime 

now = datetime.datetime.now()
timestamp = now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

MODEL_DIRECTORY = "/home/pvossler/cm_idea/white_test/"
white_train_msk_path = MODEL_DIRECTORY+"white_train_msk.csv"
# The currently available built-in models are:
# - gaussian : Multivariate Gaussian distribution
# - gmm      : Gaussian mixture model
# - mstudent : Multivariate Student's-t distribution
# - sparse   : Multivariate sparse Gaussian distribution
# model = "mixed_student"
model = "mixed_student"

# Load data
white_data = pd.read_csv("/home/pvossler/ps_job/white_clean.csv")

factor_list = pd.read_csv("/home/pvossler/ps_job/white_factor_list.csv").values.tolist()
chunk_list = pd.read_csv("/home/pvossler/ps_job/white_chunk_list.csv").values.tolist()
cat_var_idx = pd.read_csv("/home/pvossler/ps_job/white_cat_var_idx.csv").values.tolist()
factor_list = list(chain(*factor_list))
chunk_list = list(chain(*chunk_list))
cat_var_idx = list(chain(*cat_var_idx))
cat_var_idx = np.array(cat_var_idx) - 1



# Drop Y and W
X =  white_data.drop(columns=["Y","W"])


# Split train test 80-20 and save index of train data for later
X_train = X.sample(frac=0.8,random_state=200)
np.savetxt(white_train_msk_path, X_train.index, delimiter=",")

# Regularize the covariance and generate second order knockoffs
if robust_cov:
    mcd = MinCovDet().fit(X_train)
    SigmaHat = mcd.covariance_ 
else:
    SigmaHat = np.cov(X_train, rowvar=False)

# reg_val = 5e-3
SigmaHat= SigmaHat + (reg_val)*np.eye(SigmaHat.shape[0])

p = SigmaHat.shape[1]

if p > 100:
    identity_p = np.identity(SigmaHat.shape[1])
    Sigma_inv = np.linalg.solve(SigmaHat, identity_p)

    def ASDPoptim(Sigma, Sigma_inv, block_size, approx_method):
            sol = utils.asdp(Sigma, block_size, approx_method)
            s = np.diag(np.array(sol).flatten())
            C2 = 2. * s - np.dot(s, np.dot(Sigma_inv, s))        
            return(C2, s)

    C2, s = ASDPoptim(SigmaHat, Sigma_inv,50, approx_method="selfblocks")


    Ds = np.multiply(s,np.diag(SigmaHat))

    corr_g = (np.diag(SigmaHat) - np.diag(Ds)) / np.diag(SigmaHat)

else:
    second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp")

    # Measure pairwise second-order knockoff correlations
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

print(np.average(corr_g))


avg_corr = np.average(corr_g)

