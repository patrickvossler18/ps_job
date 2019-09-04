import numpy as np
from DeepKnockoffs import KnockoffMachine
from DeepKnockoffs import GaussianKnockoffs
import data
import parameters
import scipy.cluster.hierarchy as spc
from sklearn.covariance import MinCovDet, LedoitWolf
import utils
import datetime 

for p_size in [300,500]:
    print(p_size)
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%dT%H:%M:%S') + ('-%02d' % (now.microsecond / 10000))

    MODEL_DIRECTORY = "/home/pvossler/cm_idea/"

    # Number of features
    # p = 300
    p = p_size

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
    ncat = p/2
    cat_columns = np.arange(0, ncat)
    num_cuts = 4

    # Sample training data
    X_train = DataSampler.sample(n)

    SigmaHat = np.cov(X_train, rowvar=False)
    # mcd = MinCovDet().fit(X_train)
    # SigmaHat = mcd.covariance_ 
    # SigmaHat= SigmaHat + (8e-3)*np.eye(SigmaHat.shape[0])

    SigmaHat = data.cov2cor(SigmaHat)

    # Compute distance between variables based on their pairwise absolute correlations
    # with the student's t the diagonal isn't exactly 1?
    # pdist = spc.distance.pdist(1-np.abs(SigmaHat))
    # # pdist = spc.distance.squareform((np.diag(np.abs(SigmaHat))-np.abs(SigmaHat)))
    # pdist = spc.distance.pdist((np.diag(np.abs(SigmaHat))-np.abs(SigmaHat)))

    # Apply average-linkage hierarchical clustering
    # linkage = spc.linkage(pdist, method='average')
    linkage = spc.linkage(pdist,method='complete')

    pdist = spc.distance.pdist(SigmaHat)
    linkage = spc.linkage(pdist, method='complete')
    groups = spc.fcluster(linkage, 0.5 * pdist.max(), 'distance')

    # Cut the dendrogram and define the groups of variables
    # groups = spc.cut_tree(linkage, height=d_max).flatten()
    print("Divided " + str(len(groups)) + " variables into "+str(np.max(groups)+1) + " groups.")

    # Plot group sizes
    _, counts = np.unique(groups, return_counts=True)
    print("Size of largest groups: "+str(np.max(counts)))
    print("Mean groups size: "+str(np.mean(counts)))


    # Pick one representative for each cluster
    representatives = np.array([np.where(groups==g)[0][0] for g in np.arange(np.min(groups),np.max(groups))])

    # Correlation matrix for group representatives
    SigmaHat_repr = SigmaHat[representatives,:][:,representatives]

    # Print largest remaining correlation
    np.max(np.abs(SigmaHat_repr-np.eye(SigmaHat_repr.shape[0])))

    SigmaHat = SigmaHat_repr
    # TO USE LATER
    # regularizer = np.array([1e-1]*(num_cuts*ncat)+[0]*(SigmaHat.shape[1]-(num_cuts*ncat)))
    # # Initialize generator of second-order knockoffs
    # second_order = gk.GaussianKnockoffs(SigmaHat, mu=np.mean(X_train, 0), method="sdp", regularizer=regularizer)

    # Initialize generator of second-order knockoffs
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
    np.save(MODEL_DIRECTORY + 'pars' + '_p_' + str(p_size) + timestamp + '.npy', pars)

    # Where to store the machine
    checkpoint_name = MODEL_DIRECTORY + model + timestamp + '_p_' + str(p_size) 

    # Where to print progress information
    logs_name = MODEL_DIRECTORY + model + timestamp + '_p_' + str(p_size)  + "_progress.txt"

    # Initialize the machine
    machine = KnockoffMachine(pars, checkpoint_name=checkpoint_name, logs_name=logs_name)

    # Train the machine
    machine.train(X_train)

    print(timestamp)