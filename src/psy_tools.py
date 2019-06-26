import numpy as np
from datetime import datetime, timedelta
from os import makedirs
from psytrack.hyperOpt import hyperOpt
from psytrack.helper.invBlkTriDiag import getCredibleInterval
from psytrack.helper.helperFunctions import read_input
import os
import matplotlib.pyplot as plt
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api import behavior_ophys_api as boa

def get_data(experiment_id,load_dir = r'/allen/aibs/technology/nicholasc/behavior_ophys'):
    '''
        Loads data from SDK interface
        ARGS: experiment_id to load
    '''
    # full_filepath = os.path.join(load_dir, 'behavior_ophys_session_{}.nwb'.format(experiment_id))
    session = BehaviorOphysSession(api=boa.BehaviorOphysLimsApi(experiment_id)) 
    return session

def check_grace_windows(session,time_point):
    '''
        Returns true if the time point is inside the grace period after reward delivery from an earned reward or auto-reward
    '''
    hit_end_times = session.trials.stop_time[session.trials.hit].values
    hit_response_time = session.trials.response_time[session.trials.hit].values
    inside_grace_window = np.any((hit_response_time < time_point ) & (hit_end_times > time_point))
    
    auto_reward_time = session.trials.change_time[(session.trials.auto_rewarded) & (~session.trials.aborted)] + .5
    auto_end_time = session.trials.stop_time[(session.trials.auto_rewarded) & (~session.trials.aborted)]
    inside_auto_window = np.any((auto_reward_time < time_point) & (auto_end_time > time_point))
    return inside_grace_window | inside_auto_window

def format_session(session,task_zero=True,remove_consumption=True):
    '''
        Formats the data into the requirements of Psytrack
        ARGS:
            data outputed from SDK
            task_zero, if True (Default), then task regressor is (0,1), otherwise (-1,1)
            remove_consumption, if True (Default), then removes flashes following rewards   
        Returns:
            data formated for psytrack. A dictionary with key/values:
            psydata['y'] = a vector of no-licks (1) and licks(2) for each flashes
            psydata['inputs'] = a dictionary with each key an input ('random','timing', 'task', etc)
                each value has a 2D array of shape (N,M), where N is number of flashes, and M is 1 unless you want to look at history/flash interaction terms
    '''     
    # # It should be something as simple as this
    # change_flashes = session.stimlus_presentations.change_image 
    # lick_flashes = len(session.stimulus_presentations.lick_times) > 0
    if len(session.licks) < 10:
        raise Exception('Less than 10 licks in this session')   
 
    change_flashes = []
    lick_flashes = []
    omitted_flashes = []
    false_alarms = []
    hits =[]
    misses = []
    correct_rejects = []
    aborts = []
    auto_rewards=[]
    all_licks = session.licks.time.values
    for index, row in session.stimulus_presentations.iterrows():
        # Parse licks
        start_time = row.start_time
        stop_time = row.start_time + 0.75
        this_licks = np.sum((all_licks > start_time) & (all_licks < stop_time)) > 0
        # Parse change_flashes
        if index > 0:
            prev_image = session.stimulus_presentations.image_name.loc[index -1]
            this_change_flash = not ((row.image_name == prev_image) | (row.omitted) | (prev_image =='omitted'))
        else:
            this_change_flash = False
        # Parse Trial Data
        print(index)
        #trial = get_trial(session,start_time, stop_time)
        # Pack up results
        if (not check_grace_windows(session, start_time)) | (not remove_consumption) :
            lick_flashes.append(this_licks)
            change_flashes.append(this_change_flash)
            omitted_flashes.append(row.omitted)
            #aborts.append(trial.aborted)
            #if not trial.aborted:
            #    FA = trial.false_alarm
            #    MISS = trial.miss
            #    HIT = trial.hit
            #    CR = trial.correct_reject
            #    auto_reward = trial.auto_rewarde
            #    false_alarms.append(FA)
            #    misses.append(MISS)
            #    hits.append(HIT)
            #    correct_rejects.append(CR)
            #    auto_rewards.append(auto_reward)

    # map boolean vectors to the format psytrack wants
    licks = np.array([2 if x else 1 for x in lick_flashes])   
    if task_zero:
        changes = np.array([1 if x else 0 for x in change_flashes])[:,np.newaxis]
    else:
        changes = np.array([1 if x else -1 for x in change_flashes])[:,np.newaxis]
    omitted = np.array([1 if x else 0 for x in omitted_flashes])[:,np.newaxis]
    
    # Make Dictionary of inputs, and all data
    inputDict = {   'task': changes,
                    'omissions' : omitted }
    psydata = { 'y': licks, 
                'inputs':inputDict, 
                'false_alarms': false_alarms,
                'correct_reject': correct_rejects,
                'hits': hits,
                'misses':misses,
                'aborts':aborts,
                'auto_rewards':auto_rewards }
    return psydata

def get_trial(session, start_time,stop_time):
    ''' 
        returns the behavioral state for a flash
    ''' 
    trial = session.trials[(session.trials.start_time <= start_time) & (session.trials.stop_time >= stop_time)]
    return trial.loc[0]
    

def fit_weights(psydata, BIAS=True,TASK=True, OMISSIONS=False,TIMING=False):
    '''
        does weight and hyper-parameter optimization on the data in psydata
        Args: 
            psydata is a dictionary with key/values:
            psydata['y'] = a vector of no-licks (1) and licks(2) for each flashes
            psydata['inputs'] = a dictionary with each key an input ('random','timing', 'task', etc)
                each value has a 2D array of shape (N,M), where N is number of flashes, and M is 1 unless you want to look at history/flash interaction terms

        RETURNS:
        hyp
        evd
        wMode
        hess
    '''
    weights = {}
    if BIAS: weights['bias'] = 1
    if TASK: weights['task'] = 1
    if OMISSIONS: weights['omissions'] = 1
    if TIMING: weights['timing'] = 1

    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
            'sigma':[2**-4.]*K,
            'sigDay': None}
    optList=['sigma']
    hyp,evd,wMode,hess =hyperOpt(psydata,hyper,weights, optList)
    credibleInt = getCredibleInterval(hess)
    return hyp, evd, wMode, hess, credibleInt, weights

def compute_ypred(psydata, wMode, weights):
    g = read_input(psydata, weights)
    gw = np.sum(g*wMode.T,axis=1)
    pR = 1/(1+np.exp(-gw))
    return pR

def transform(series):
    '''
        passes the series through the logistic function
    '''
    return 1/(1+np.exp(-(series)))

def get_flash_index(session, time_point):
    '''
        Returns the flash index of a time point
    '''
    return np.where(session.stimulus_presentations.start_time.values < time_point)[0][-1]

def moving_mean(values, window):
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm

def plot_weights(session, wMode,weights,psydata,errorbar=None, ypred=None,START=0, END=0,remove_consumption=True):
    K,N = wMode.shape    
    if START <0: START = 0
    if START > N: raise Exception(" START > N")
    if END <=0: END = N
    if END > N: END = N
    if START >= END: raise Exception("START >= END")

    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
   
    my_colors=['blue','green','purple']  

    if not (type(ypred) == type(None)):
        fig,ax = plt.subplots(nrows=4,ncols=1, figsize=(10,10))
        ax[3].plot(ypred, 'k',alpha=0.3,label='Full Model')
        ax[3].plot(moving_mean(psydata['y']-1,25), 'b',alpha=0.5,label='data (n=25)')
        ax[3].set_ylim(0,1)
        ax[3].set_ylabel('Lick Prob',fontsize=12)
        ax[3].set_xlabel('Flash #',fontsize=12)
        ax[3].set_xlim(START,END)
        ax[3].legend()
        ax[3].tick_params(axis='both',labelsize=12)
    else:
        fig,ax = plt.subplots(nrows=3,ncols=1, figsize=(10,10))
    for i in np.arange(0, len(weights_list)):
        ax[0].plot(wMode[i,:], linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])        
        ax[0].fill_between(np.arange(len(wMode[i])), wMode[i,:]-2*errorbar[i], 
            wMode[i,:]+2*errorbar[i],facecolor=my_colors[i], alpha=0.2)    
        ax[1].plot(transform(wMode[i,:]), linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])
        ax[1].fill_between(np.arange(len(wMode[i])), transform(wMode[i,:]-2*errorbar[i]), 
            transform(wMode[i,:]+2*errorbar[i]),facecolor=my_colors[i], alpha=0.2)                  

    ax[0].plot([0,np.shape(wMode)[1]], [0,0], 'k--',alpha=0.2)
    ax[0].set_ylabel('Weight',fontsize=12)
    ax[0].set_xlabel('Flash #',fontsize=12)
    ax[0].set_xlim(START,END)
    ax[0].legend()
    ax[0].tick_params(axis='both',labelsize=12)
 
    ax[1].set_ylim(0,1)
    ax[1].set_ylabel('Lick Prob',fontsize=12)
    ax[1].set_xlabel('Flash #',fontsize=12)
    ax[1].set_xlim(START,END)
    ax[1].legend()
    ax[1].tick_params(axis='both',labelsize=12)

    first_start = session.trials.loc[0].start_time
    jitter = 0.025
    for index, row in session.trials.iterrows(): 
        if row.hit:
            ax[2].plot(get_flash_index(session, row.change_time), 1+np.random.randn()*jitter, 'bo',alpha=0.2)
        elif row.miss:
            ax[2].plot(get_flash_index(session, row.change_time), 1.5+np.random.randn()*jitter, 'ro',alpha = 0.2)   
        elif row.false_alarm:
            ax[2].plot(get_flash_index(session, row.change_time), 2.5+np.random.randn()*jitter, 'ko',alpha = 0.2)
        elif row.correct_reject & (not row.aborted):
            ax[2].plot(get_flash_index(session, row.change_time), 2+np.random.randn()*jitter, 'co',alpha = 0.2)   
        elif row.aborted:
            if len(row.lick_times) >= 1:
                ax[2].plot(get_flash_index(session, row.lick_times[0]), 3+np.random.randn()*jitter, 'ko',alpha=0.2)   
            else:  
                ax[2].plot(get_flash_index(session, row.start_time), 3+np.random.randn()*jitter, 'ko',alpha=0.2)  
        else:
            raise Exception('Trial had no classification')
        if row.auto_rewarded & (not row.aborted):
            ax[2].plot(get_flash_index(session, row.change_time), 3.5+np.random.randn()*jitter, 'go',alpha=0.2)    
 
    ax[2].set_yticks([1,1.5,2,2.5,3,3.5])
    ax[2].set_yticklabels(['hits','miss','CR','FA','abort','auto'],{'fontsize':12})
    #ax[2].set_xlim(START,END)
    ax[2].set_xlabel('Flash # (unaligned!)',fontsize=12)
    ax[2].tick_params(axis='both',labelsize=12)
    plt.tight_layout()



def check_lick_alignment(session, psydata):
    plt.figure(figsize=(10,5))
    plt.plot(session.stimulus_presentations.start_time.values,psydata['y']-1, 'ko-')
    all_licks = session.licks
    for index, lick in all_licks.iterrows():
        plt.plot([lick.time, lick.time], [0.9, 1.1], 'r')
    plt.xlabel('time (s)')
    for index, row in session.trials.iterrows():
        if row.hit:
            plt.plot(row.change_time, 1.2, 'bo')
        elif row.miss:
            plt.plot(row.change_time, 1.25, 'gx')   
        elif row.false_alarm:
            plt.plot(row.change_time, 1.3, 'ro')
        elif row.correct_reject:
            plt.plot(row.change_time, 1.35, 'cx')   
        elif row.aborted:
            if len(row.lick_times) >= 1:
                plt.plot(row.lick_times[0], 1.4, 'kx')   
            else:  
                plt.plot(row.start_time, 1.4, 'kx')  
        else:
            raise Exception('Trial had no classification')
   


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




