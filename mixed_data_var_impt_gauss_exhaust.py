import numpy as np
import pandas as pd
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import data
import parameters
from sklearn.covariance import MinCovDet, LedoitWolf
import utils
import datetime 



# Number of features
p_list = [50]
for p in p_list:
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

    MODEL_DIRECTORY = "/home/pvossler/cm_idea/exhaustive_test/exhaustive_test_2e3/exhaustive_test"
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
    X_train.iloc[:, cat_columns] = X_train.iloc[:, cat_columns].apply(lambda x: pd.qcut(x, 4, retbins=False, labels=False), axis=0).astype(str)
    X_train_dums = pd.get_dummies(X_train.iloc[:, cat_columns], prefix=X_train.iloc[:, cat_columns].columns.values.astype(str).tolist())
    X_train = pd.concat([X_train_dums.reset_index(drop=True), X_train.drop(cat_columns, axis=1).reset_index(drop=True)], axis=1)

    # SigmaHat = np.cov(X_train, rowvar=False)
    mcd = MinCovDet().fit(X_train)
    SigmaHat = mcd.covariance_ 

    SigmaHat= SigmaHat + (2e-3)*np.eye(SigmaHat.shape[0])

    # second_order = gk.GaussianKnockoffs(SigmaHat_mcd, mu=np.mean(X_train, 0), method="sdp", regularizer=1e-1)
    # second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=1e-1)

    second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp")

    # Measure pairwise second-order knockoff correlations
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)

    print(np.average(corr_g))

    print(np.average(corr_g[1:(num_cuts*ncat)]))
    print(np.average(corr_g[((num_cuts*ncat)+1):((num_cuts*ncat)+ int(p/2))]))

    training_params = parameters.GetTrainingHyperParams(model)
    # training_params['LAMBDA'] = 0.0078
    # training_params['DELTA'] = 0.0078
    p = X_train.shape[1]
    print(p)
    print(X_train.shape)
    chunk_list = [num_cuts] * (ncat)
    a = np.linspace(0.01,0.001,num=5)
    b = np.linspace(0.01,0.001,num=5)
    param_combos = [(x,y) for x in a for y in b]

    for combo in param_combos:
        # Set the parameters for training deep knockoffs
        pars = dict()
        # Number of epochs
        pars['epochs'] = 50
        # Number of iterations over the full data per epoch
        pars['epoch_length'] = 50
        # Data type, either "continuous" or "binary"
        pars['family'] = "continuous"
        # Dimensions of the data
        pars['p'] = p
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
        np.save(MODEL_DIRECTORY + 'pars' + "_" + par_lambda + "_" + par_delta + '_p_' + str(p_size) + timestamp + '.npy', pars)

        # Where to store the machine
        checkpoint_name = MODEL_DIRECTORY + model + "_" + par_lambda + "_" + par_delta + timestamp + '_p_' + str(p_size) 

        # Where to print progress information
        logs_name = MODEL_DIRECTORY + model + "_" + par_lambda + "_" + par_delta+ timestamp + '_p_' + str(p_size)  + "_progress.txt"

        # Initialize the machine
        machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)

        # Train the machine
        machine.train(X_train)

        print(timestamp)