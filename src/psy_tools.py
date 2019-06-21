import numpy as np
from datetime import datetime, timedelta
from os import makedirs

def generateSim_VB(K=4,
                N=64000,
                hyper={},
                boundary=4.0,
                iterations=20,
                seed=None,
                savePath=None):
    """Simulates weights, in addition to inputs and multiple realizations
    of responses. Simulation data is either saved to a file or returned
    directly.
    Args:
        K : int, number of weights to simulate
        N : int, number of trials to simulate
        hyper : dict, hyperparameters and initial values used to construct the
            prior. Default is none, can include sigma, sigInit, sigDay
        boundary : float, weights are reflected from this boundary
            during simulation, is a symmetric +/- boundary
        iterations : int, # of behavioral realizations to simulate,
            same input and weights can render different choice due
            to probabilistic model, iterations are saved in 'all_Y'
        seed : int, random seed to make random simulations reproducible
        savePath : str, if given creates a folder and saves simulation data
            in a file; else data is returned
    Returns:
        save_path | (if savePath) : str, the name of the folder+file where
            simulation data was saved in the local directory
        save_dict | (if no SavePath) : dict, contains all relevant info
            from the simulation 
    """

    # Reproducability
    np.random.seed(seed)

    # Supply default hyperparameters if necessary
    sigmaDefault = 2**np.random.choice([-4.0, -5.0, -6.0, -7.0, -8.0], size=K)
    if "sigma" not in hyper:
        sigma = sigmaDefault
    elif hyper["sigma"] is None:
        sigma = sigmaDefault
    elif np.isscalar(hyper["sigma"]):
        sigma = np.array([hyper["sigma"]] * K)
    elif ((type(hyper["sigma"]) in [np.ndarray, list]) and
          (len(hyper["sigma"]) != K)):
        sigma = hyper["sigma"]
    else:
        raise Exception(
            "hyper['sigma'] must be either a scalar or a list or array of len K"
        )

    sigInitDefault = np.array([4.0] * K)
    if "sigInit" not in hyper:
        sigInit = sigInitDefault
    elif hyper["sigInit"] is None:
        sigInit = sigInitDefault
    elif np.isscalar(hyper["sigInit"]):
        sigInit = np.array([hyper["sigInit"]] * K)
    elif (type(hyper["sigInit"]) in [np.ndarray, list]) and (len(hyper["sigInit"]) != K):
        sigInit = hyper["sigInit"]
    else:
        raise Exception("hyper['sigInit'] must be either a scalar or \
            a list or array of len K")

    # sigDay not yet supported!
    if "sigDay" in hyper and hyper["sigDay"] is not None:
        raise Exception("sigDay not yet supported, please omit from hyper")

    # -------------
    # Simulation
    # -------------

    # Simulate inputs
    X = np.random.normal(size=(N, K))
    X[:,0] = 1
    X[:,1] = np.abs(np.sin(np.arange(0,N,3.14/10)))[0:N]
    X[:,2] = 0   
    X[np.random.normal(size=(N,)) > 1,2] = 1

    # Simulate weights
    E = np.zeros((N, K))
    E[0] = np.random.normal(scale=sigInit, size=K)
    E[1:] = np.random.normal(scale=sigma, size=(N - 1, K))
    W = np.cumsum(E, axis=0)

    # Impose a ceiling and floor boundary on W
    for i in range(len(W.T)):
        cross = (W[:, i] < -boundary) | (W[:, i] > boundary)
        while cross.any():
            ind = np.where(cross)[0][0]
            if W[ind, i] < -boundary:
                W[ind:, i] = -2 * boundary - W[ind:, i]
            else:
                W[ind:, i] = 2 * boundary - W[ind:, i]
            cross = (W[:, i] < -boundary) | (W[:, i] > boundary)

    # Save data
    save_dict = {
        "sigInit": sigInit,
        "sigma": sigma,
        "seed": seed,
        "W": W,
        "X": X,
        "K": K,
        "N": N,
    }

    # Simulate behavioral realizations in advance
    pR = 1.0 / (1.0 + np.exp(-np.sum(X * W, axis=1)))

    all_simy = []
    for i in range(iterations):
        sim_y = (pR > np.random.rand(
            len(pR))).astype(int) + 1  # 1 for L, 2 for R
        all_simy += [sim_y]

    # Update saved data to include behavior
    save_dict.update({"all_Y": all_simy})

    # Save & return file path OR return simulation data
    if savePath is not None:
        # Creates unique file name from current datetime
        folder = datetime.now().strftime("%Y%m%d_%H%M%S") + savePath
        makedirs(folder)

        fullSavePath = folder + "/sim.npz"
        np.savez_compressed(fullSavePath, save_dict=save_dict)

        return fullSavePath

    else:
        return save_dict
