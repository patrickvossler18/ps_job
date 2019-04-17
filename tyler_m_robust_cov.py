import numpy as np


def iteration(sigma_hat):
    sigma_hat_inv = np.linalg.inv(sigma_hat)
    mult_res = np.sum(np.apply_along_axis(multiply,1,x_normed, sigma_hat_inv),axis=0)
    sigma_hat = (p/n)  * mult_res
    return(sigma_hat)

# Normalize samples
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def multiply(x,sigma_hat_inv):
    x = np.expand_dims(x, axis=1)
    return((x @ x.T) / (x.T @ sigma_hat_inv @ x))


def fixedp(x_normed,x0,tol=10e-5,maxiter=100):
    """ Fixed point algorithm """
    e = 1
    itr = 0
    # xp = []
    x_normed = x_normed
    while(e > tol and itr < maxiter):
        print(itr)
        x = iteration(x0)      # fixed point equation
        e = np.linalg.norm(x0-x) # error at the current step
        x0 = x
        # xp.append(x0)  # save the solution of the current step
        itr = itr + 1
    return x



def getTylerM(X_train):
    x_normed = X_train.apply(normalize,axis=1).values
    n, p = X_train.shape
    sigma_start = np.identity(p)
    robust_cov = fixedp(x_normed,sigma_start)
    return(robust_cov)




# def iteration_highdim(sigma_hat):
#     sigma_hat_inv = np.linalg.inv(sigma_hat)
#     sigma_tilde = ((1-rho) * (p/n)  *\
#             np.sum(np.apply_along_axis(multiply,1,x_normed, sigma_hat_inv))) +\
#             rho *np.identity(p)
#     return sigma_tilde/ (np.trace(sigma_tilde)/p)




# # trace-normalized sample covariance
# lw = LedoitWolf().fit(X_train)
# R_hat = np.trace(lw.covariance_) / n * lw.covariance_
# R_hat = p/n * (np.cov(X_train,rowvar=False) + (2)*np.eye(X_train.shape[1]))
# rho = ((p**2) + ((1- 2/p) * np.trace(np.linalg.matrix_power(R_hat,2))))/\
#     ((p**2 - n*p - 2*n) + (n + 1 + (2*(n-1)/p)) * np.trace(np.linalg.matrix_power(R_hat,2)))

# rho = ((p**2) + ((1- 2/p) * np.trace(R_hat**2)))/((p**2 - n*p - 2*n) + (n + 1 + (2*(n-1)/p)) * np.trace(R_hat**2))

# sigma_hat_inv = np.linalg.inv(sigma_start)


# def fixedp_highdim(x_normed,x0,tol=10e-5,maxiter=100):
#     """ Fixed point algorithm """
#     e = 1
#     itr = 0
#     xp = []
#     x_normed = x_normed
#     while(e > tol and itr < maxiter):
#         print(itr)
#         x = iteration_highdim(x0)      # fixed point equation
#         e = np.linalg.norm(x0-x) # error at the current step
#         x0 = x
#         xp.append(x0)  # save the solution of the current step
#         itr = itr + 1
#     return x,xp

# test = fixedp_highdim(x_normed,sigma_start)