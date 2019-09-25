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



p = int(p_arg)
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
model = "gaussian"
distribution_params = parameters.GetDistributionParams(model, p)

# Initialize the data generator
DataSampler = data.DataSampler(distribution_params)

# Number of training examples
n = 1000


# USE THIS FOR JUST K DUMMY VARIABLES
X_train = pd.DataFrame(DataSampler.sample(n))

if robust_cov:
    mcd = MinCovDet().fit(X_train)
    SigmaHat = mcd.covariance_ 
else:
    SigmaHat = np.cov(X_train, rowvar=False)

# reg_val = 5e-3
SigmaHat= SigmaHat + (reg_val)*np.eye(SigmaHat.shape[0])

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

print(np.average(corr_g[1:(np.sum(num_cuts)-1)]))
print(np.average(corr_g[((np.sum(num_cuts)-1)+1):]))

avg_corr = np.average(corr_g)

