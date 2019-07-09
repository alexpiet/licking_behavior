import numpy as np
from datetime import datetime, timedelta
import os
from os import makedirs
import copy
import pickle
import matplotlib.pyplot as plt
from psytrack.hyperOpt import hyperOpt
from psytrack.helper.invBlkTriDiag import getCredibleInterval
from psytrack.helper.helperFunctions import read_input
from psytrack.helper.crossValidation import Kfold_crossVal
from psytrack.helper.crossValidation import Kfold_crossVal_check
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api import behavior_ophys_api as boa
from sklearn.linear_model import LinearRegression
from sklearn.cluster import k_means


def load(filepath):
    filetemp = open(filepath,'rb')
    data    = pickle.load(filetemp)
    filetemp.close()
    return data

def save(filepath, variables):
    file_temp = open(filepath,'wb')
    pickle.dump(variables, file_temp)
    file_temp.close()

def get_data(experiment_id,load_dir = r'/allen/aibs/technology/nicholasc/behavior_ophys'):
    '''
        Loads data from SDK interface
        ARGS: experiment_id to load
    '''
    # full_filepath = os.path.join(load_dir, 'behavior_ophys_session_{}.nwb'.format(experiment_id))
    api=boa.BehaviorOphysLimsApi(experiment_id)
    session = BehaviorOphysSession(api) 
    session.metadata['stage'] = api.get_task_parameters()['stage']
    return session

def get_stage(experiment_id):
    api=boa.BehaviorOphysLimsApi(experiment_id)
    return api.get_task_parameters()['stage']

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

def format_all_sessions(all_flash_df, remove_consumption=True):
    change_flashes = []
    omitted_flashes = []
    omitted_1_flashes = []
    timing_flashes4 = []
    timing_flashes5 = []
    last_omitted = False
    lick_flashes = all_flash_df.lick_bool.values
    prev_image = all_flash_df.loc[0].image_name
    num_since_lick = 0
    last_num_since_lick =0
    for index, row in all_flash_df.iterrows():
        # Parse licks
        start_time = row.start_time
        stop_time = row.start_time + 0.75
        # Parse timing drive
        last_num_since_lick = num_since_lick
        this_licks = row.lick_bool
        if this_licks:
            num_since_lick = 0
        else:
            num_since_lick +=1
  
        # Parse change_flashes
        if index > 0:
            this_change_flash = not ((row.image_name == prev_image) | (row.omitted) | (prev_image =='omitted'))
        else:
            this_change_flash = False
        prev_image = row.image_name
        change_flashes.append(this_change_flash)
        omitted_flashes.append(row.omitted)
        omitted_1_flashes.append(last_omitted)
        last_omitted = row.omitted
        timing_flashes4.append(timing_curve4(last_num_since_lick))
        timing_flashes5.append(timing_curve5(last_num_since_lick))
    # map boolean vectors to the format psytrack wants
    licks       = np.array([2 if x else 1 for x in lick_flashes])   
    changes0    = np.array([1 if x else 0 for x in change_flashes])[:,np.newaxis]
    changes1    = np.array([1 if x else -1 for x in change_flashes])[:,np.newaxis]
    changesCR   = np.array([0 if x else -1 for x in change_flashes])[:,np.newaxis]
    omitted     = np.array([1 if x else 0 for x in omitted_flashes])[:,np.newaxis]
    omitted1    = np.array([1 if x else 0 for x in omitted_1_flashes])[:,np.newaxis]
    timing4     = np.array(timing_flashes4)[:,np.newaxis]
    timing5     = np.array(timing_flashes5)[:,np.newaxis] 
    session_dex = np.unique(all_flash_df.session_index.values)
    dayLength = []
    for dex in session_dex:
        dayLength.append(np.sum(all_flash_df.session_index.values == dex))

    inputDict = {   'task0': changes0,
                    'task1': changes1,
                    'taskCR': changesCR,
                    'omissions' : omitted,
                    'omissions1' : omitted1,
                    'timing4': timing4,
                    'timing5': timing5 }
    psydata = { 'y': licks, 
                'inputs':inputDict, 
                #'false_alarms': false_alarms,
                #'correct_reject': correct_rejects,
                #'hits': hits,
                #'misses':misses,
                #'aborts':aborts,
                #'auto_rewards':auto_rewards,
                #'start_times':start_times,
                'dayLength':np.array(dayLength)  }
    return psydata



def format_session(session,remove_consumption=True):
    '''
        Formats the data into the requirements of Psytrack
        ARGS:
            data outputed from SDK
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
    omitted_1_flashes = []
    timing_flashes4 = []
    timing_flashes5 = []
    false_alarms = []
    hits =[]
    misses = []
    correct_rejects = []
    aborts = []
    auto_rewards=[]
    start_times=[]
    all_licks = session.licks.time.values
    num_since_lick = 0
    last_num_since_lick =0
    last_omitted = False
    for index, row in session.stimulus_presentations.iterrows():
        # Parse licks
        start_time = row.start_time
        stop_time = row.start_time + 0.75
        this_licks = np.sum((all_licks > start_time) & (all_licks < stop_time)) > 0
        # Parse timing drive
        last_num_since_lick = num_since_lick
        if this_licks:
            num_since_lick = 0
        else:
            num_since_lick +=1
        
        # Parse change_flashes
        if index > 0:
            prev_image = session.stimulus_presentations.image_name.loc[index -1]
            this_change_flash = not ((row.image_name == prev_image) | (row.omitted) | (prev_image =='omitted'))
        else:
            this_change_flash = False
        # Parse Trial Data 
        # Pack up results
        if (not check_grace_windows(session, start_time)) | (not remove_consumption) :
            trial = get_trial(session,start_time, stop_time)
            lick_flashes.append(this_licks)
            change_flashes.append(this_change_flash)
            omitted_flashes.append(row.omitted)
            omitted_1_flashes.append(last_omitted)
            aborts.append(trial['aborted']) 
            false_alarms.append(trial['false_alarm'])
            misses.append(trial['miss'])
            hits.append(trial['hit'])
            correct_rejects.append(trial['correct_reject'])
            auto_rewards.append(trial['auto_rewarded'])
            start_times.append(start_time)
            timing_flashes4.append(timing_curve4(last_num_since_lick))
            timing_flashes5.append(timing_curve5(last_num_since_lick))
        last_omitted = row.omitted
    # map boolean vectors to the format psytrack wants
    licks       = np.array([2 if x else 1 for x in lick_flashes])   
    changes0    = np.array([1 if x else 0 for x in change_flashes])[:,np.newaxis]
    changes1    = np.array([1 if x else -1 for x in change_flashes])[:,np.newaxis]
    changesCR   = np.array([0 if x else -1 for x in change_flashes])[:,np.newaxis]
    omitted     = np.array([1 if x else 0 for x in omitted_flashes])[:,np.newaxis]
    omitted1    = np.array([1 if x else 0 for x in omitted_1_flashes])[:,np.newaxis]
    timing4     = np.array(timing_flashes4)[:,np.newaxis]
    timing5     = np.array(timing_flashes5)[:,np.newaxis] 
    # Make Dictionary of inputs, and all data
    inputDict = {   'task0': changes0,
                    'task1': changes1,
                    'taskCR': changesCR,
                    'omissions' : omitted,
                    'omissions1' : omitted1,
                    'timing4': timing4,
                    'timing5': timing5 }
    psydata = { 'y': licks, 
                'inputs':inputDict, 
                'false_alarms': false_alarms,
                'correct_reject': correct_rejects,
                'hits': hits,
                'misses':misses,
                'aborts':aborts,
                'auto_rewards':auto_rewards,
                'start_times':start_times }
    return psydata

def timing_curve4(num_flashes):
    '''
        Defines a timing function that maps the number of flashes from the last lick to the timing drive to lick on this flash
        num_flashes = 0 means I licked on this flash
        num_flashes = 1 means I licked on the last flash, but not this one. 
    '''
    if num_flashes < 0:
        raise Exception ('Timing cant be negative')
    elif num_flashes == 0:
        return 0
    elif num_flashes < 4:
        return -1
    elif num_flashes == 4:
        return -.5
    elif num_flashes == 5:
        return 0.5
    else:
        return 1

def timing_curve5(num_flashes):
    '''
        Defines a timing function that maps the number of flashes from the last lick to the timing drive to lick on this flash
        num_flashes = 0 means I licked on this flash
        num_flashes = 1 means I licked on the last flash, but not this one. 
    '''
    if num_flashes < 0:
        raise Exception ('Timing cant be negative')
    elif num_flashes == 0:
        return 0
    elif num_flashes < 5:
        return -1
    elif num_flashes == 5:
        return -.5
    elif num_flashes == 6:
        return 0.5
    else:
        return 1



def get_trial(session, start_time,stop_time):
    ''' 
        returns the behavioral state for a flash
    '''
    if start_time > stop_time:
        raise Exception('Start time cant be later than stop time') 
    trial = session.trials[(session.trials.start_time <= start_time) & (session.trials.stop_time >= stop_time)]
    if len(trial) == 0:
        trial = session.trials[(session.trials.start_time <= start_time) & (session.trials.stop_time+0.75 >= stop_time)]
        if len(trial) == 0:
            labels = {  'aborted':False,
                'hit': False,
                'miss': False,
                'false_alarm': False,
                'correct_reject': False,
                'auto_rewarded': False  }
            return labels
        else:
            trial = trial.iloc[0]
    else:
        trial = trial.iloc[0]

    labels = {  'aborted':trial.aborted,
                'hit': trial.hit,
                'miss': trial.miss,
                'false_alarm': trial.false_alarm,
                'correct_reject': trial.correct_reject & (not trial.aborted),
                'auto_rewarded': trial.auto_rewarded & (not trial.aborted)  }
    if trial.hit:
        labels['hit'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    if trial.miss:
        labels['miss'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    if trial.false_alarm:
        labels['false_alarm'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    if (trial.correct_reject) &  (not trial.aborted):
        labels['correct_reject'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    if trial.aborted:
        if len(trial.lick_times) >= 1:
            labels['aborted'] = (trial.lick_times[0] >= start_time ) & (trial.lick_times[0] < stop_time)
        else:
            labels['aborted'] = (trial.start_time >= start_time ) & (trial.start_time < stop_time)
    if trial.auto_rewarded & (not trial.aborted):
            labels['auto_rewarded'] = (trial.change_time >= start_time) & (trial.change_time < stop_time )
    return labels
    
def fit_weights(psydata, BIAS=True,TASK0=True, TASK1=False,TASKCR = False, OMISSIONS=False,OMISSIONS1=False,TIMING4=False,TIMING5=False,fit_overnight=False):
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
    if TASK0: weights['task0'] = 1
    if TASK1: weights['task1'] = 1
    if TASKCR: weights['taskCR'] = 1
    if OMISSIONS: weights['omissions'] = 1
    if OMISSIONS1: weights['omissions1'] = 1
    if TIMING4: weights['timing4'] = 1
    if TIMING5: weights['timing5'] = 1
    print(weights)

    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
            'sigma':[2**-4.]*K,
            'sigDay': 2**4}
    optList=['sigma']
    hyp,evd,wMode,hess =hyperOpt(psydata,hyper,weights, optList)
    credibleInt = getCredibleInterval(hess)
    return hyp, evd, wMode, hess, credibleInt, weights

def compute_ypred(psydata, wMode, weights):
    g = read_input(psydata, weights)
    gw = g*wMode.T
    total_gw = np.sum(g*wMode.T,axis=1)
    pR = 1/(1+np.exp(-total_gw))
    pR_each = 1/(1+np.exp(-gw))
    return pR, pR_each

def transform(series):
    '''
        passes the series through the logistic function
    '''
    return 1/(1+np.exp(-(series)))

def get_flash_index_session(session, time_point):
    '''
        Returns the flash index of a time point
    '''
    return np.where(session.stimulus_presentations.start_time.values < time_point)[0][-1]

def get_flash_index(psydata, time_point):
    '''
        Returns the flash index of a time point
    '''
    if time_point > psydata['start_times'][-1] + 0.75:
        return np.nan
    return np.where(np.array(psydata['start_times']) < time_point)[0][-1]


def moving_mean(values, window):
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm

def plot_weights(session, wMode,weights,psydata,errorbar=None, ypred=None,START=0, END=0,remove_consumption=True,validation=True,session_labels=None, seedW = None,ypred_each = None,filename=None,cluster_labels=None):
    K,N = wMode.shape    
    if START <0: START = 0
    if START > N: raise Exception(" START > N")
    if END <=0: END = N
    if END > N: END = N
    if START >= END: raise Exception("START >= END")

    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
   
    my_colors=['blue','green','purple','red']  
    if 'dayLength' in psydata:
        dayLength = np.concatenate([[0],np.cumsum(psydata['dayLength'])])
    else:
        dayLength = []

    cluster_ax = 3
    if (not (type(ypred) == type(None))) & validation:
        fig,ax = plt.subplots(nrows=4,ncols=1, figsize=(10,10))
        ax[3].plot(ypred, 'k',alpha=0.3,label='Full Model')
        if not( type(ypred_each) == type(None)):
            for i in np.arange(0, len(weights_list)):
                ax[3].plot(ypred_each[:,i], linestyle="-", lw=3, alpha = 0.3,color=my_colors[i],label=weights_list[i])        
        ax[3].plot(moving_mean(psydata['y']-1,25), 'b',alpha=0.5,label='data (n=25)')
        ax[3].set_ylim(0,1)
        ax[3].set_ylabel('Lick Prob',fontsize=12)
        ax[3].set_xlabel('Flash #',fontsize=12)
        ax[3].set_xlim(START,END)
        ax[3].legend()
        ax[3].tick_params(axis='both',labelsize=12)
    elif validation:
        fig,ax = plt.subplots(nrows=3,ncols=1, figsize=(10,8))
        cluster_ax = 2
    elif (not (type(cluster_labels) == type(None))):
        fig,ax = plt.subplots(nrows=3,ncols=1, figsize=(10,8))
        cluster_ax = 2
    else:
        fig,ax = plt.subplots(nrows=2,ncols=1, figsize=(10,6)  )
    if (not (type(cluster_labels) == type(None))):
        cp = np.where(~(np.diff(cluster_labels) == 0))[0]
        cp = np.concatenate([[0], cp, [len(cluster_labels)]])
        cluster_colors = ['r','b','g','c','m','k','y']
        for i in range(0, len(cp)-1):
            ax[cluster_ax].axvspan(cp[i],cp[i+1],color=cluster_colors[cluster_labels[cp[i]+1]], alpha=0.1)
    for i in np.arange(0, len(weights_list)):
        ax[0].plot(wMode[i,:], linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])        
        ax[0].fill_between(np.arange(len(wMode[i])), wMode[i,:]-2*errorbar[i], 
            wMode[i,:]+2*errorbar[i],facecolor=my_colors[i], alpha=0.2)    
        ax[1].plot(transform(wMode[i,:]), linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])
        ax[1].fill_between(np.arange(len(wMode[i])), transform(wMode[i,:]-2*errorbar[i]), 
            transform(wMode[i,:]+2*errorbar[i]),facecolor=my_colors[i], alpha=0.2)                  
        if not (type(seedW) == type(None)):
            ax[0].plot(seedW[i,:], linestyle="--", lw=2, color=my_colors[i], label= "seed "+weights_list[i])
            ax[1].plot(transform(seedW[i,:]), linestyle="--", lw=2, color=my_colors[i], label= "seed "+weights_list[i])
    ax[0].plot([0,np.shape(wMode)[1]], [0,0], 'k--',alpha=0.2)
    ax[0].set_ylabel('Weight',fontsize=12)
    ax[0].set_xlabel('Flash #',fontsize=12)
    ax[0].set_xlim(START,END)
    ax[0].legend()
    ax[0].tick_params(axis='both',labelsize=12)
    for i in np.arange(0, len(dayLength)-1):
        ax[0].axvline(dayLength[i],color='k',alpha=0.2)
        if not type(session_labels) == type(None):
            ax[0].text(dayLength[i],ax[0].get_ylim()[1], session_labels[i][0:10],rotation=25)
    ax[1].set_ylim(0,1)
    ax[1].set_ylabel('Lick Prob',fontsize=12)
    ax[1].set_xlabel('Flash #',fontsize=12)
    ax[1].set_xlim(START,END)
    ax[1].legend(loc='upper right')
    ax[1].tick_params(axis='both',labelsize=12)
    for i in np.arange(0, len(dayLength)-1):
        ax[1].plot([dayLength[i], dayLength[i]],[0,1], 'k-',alpha=0.2)

    if validation:
        first_start = session.trials.loc[0].start_time
        jitter = 0.025   
        for i in np.arange(0, len(psydata['hits'])):
            if psydata['hits'][i]:
                ax[2].plot(i, 1+np.random.randn()*jitter, 'bo',alpha=0.2)
            elif psydata['misses'][i]:
                ax[2].plot(i, 1.5+np.random.randn()*jitter, 'ro',alpha = 0.2)   
            elif psydata['false_alarms'][i]:
                ax[2].plot(i, 2.5+np.random.randn()*jitter, 'ko',alpha = 0.2)
            elif psydata['correct_reject'][i] & (not psydata['aborts'][i]):
                ax[2].plot(i, 2+np.random.randn()*jitter, 'co',alpha = 0.2)   
            elif psydata['aborts'][i]:
                ax[2].plot(i, 3+np.random.randn()*jitter, 'ko',alpha=0.2)  
            if psydata['auto_rewards'][i] & (not psydata['aborts'][i]):
                ax[2].plot(i, 3.5+np.random.randn()*jitter, 'go',alpha=0.2)    
    
        ax[2].set_yticks([1,1.5,2,2.5,3,3.5])
        ax[2].set_yticklabels(['hits','miss','CR','FA','abort','auto'],{'fontsize':12})
        ax[2].set_xlim(START,END)
        ax[2].set_xlabel('Flash #',fontsize=12)
        ax[2].tick_params(axis='both',labelsize=12)

    plt.tight_layout()
    if not (type(filename) == type(None)):
        plt.savefig(filename+"_weights.png")
    

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


def sample_model(psydata):
    bootdata = copy.copy(psydata)    
    if not ('ypred' in bootdata):
        raise Exception('You need to compute y-prediction first')
    temp = np.random.random(np.shape(bootdata['ypred'])) < bootdata['ypred']
    licks = np.array([2 if x else 1 for x in temp])   
    bootdata['y'] = licks
    return bootdata


def bootstrap_model(psydata, ypred, weights,seedW,plot_this=True):
    psydata['ypred'] =ypred
    bootdata = sample_model(psydata)
    bK = np.sum([weights[i] for i in weights.keys()])
    bhyper = {'sigInit': 2**4.,
        'sigma':[2**-4.]*bK,
        'sigDay': 2**4}
    boptList=['sigma']
    bhyp,bevd,bwMode,bhess =hyperOpt(bootdata,bhyper,weights, boptList)
    bcredibleInt = getCredibleInterval(bhess)
    if plot_this:
        plot_weights(None, bwMode, weights, bootdata, errorbar=bcredibleInt, validation=False,seedW =seedW )
    return (bootdata, bhyp, bevd, bwMode, bhess, bcredibleInt)

def bootstrap(numboots, psydata, ypred, weights, seedW, plot_each=False):
    boots = []
    for i in np.arange(0,numboots):
        print(i)
        boot = bootstrap_model(psydata, ypred, weights, seedW,plot_this=plot_each)
        boots.append(boot)
    return boots

def plot_bootstrap(boots, hyp, weights, seedW, credibleInt,filename=None):
    plot_bootstrap_recovery_prior(boots,hyp, weights,filename)
    plot_bootstrap_recovery_weights(boots,hyp, weights,seedW,credibleInt,filename)


def plot_bootstrap_recovery_prior(boots,hyp,weights,filename):
    fig,ax = plt.subplots(figsize=(3,4))
    my_colors=['blue','green','purple','red']  
    plt.yscale('log')
    plt.ylim(0.001, 20)
    ax.set_xticks(np.arange(0,len(hyp['sigma'])))
    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
    ax.set_xticklabels(weights_list)
    plt.ylabel('Smoothing Prior, $\sigma$')
    for boot in boots:
        plt.plot(boot[1]['sigma'], 'kx',alpha=0.5)
    for i in np.arange(0, len(hyp['sigma'])):
        plt.plot(i,hyp['sigma'][i], 'o', color=my_colors[i])

    plt.tight_layout()
    if not (type(filename) == type(None)):
        plt.savefig(filename+"_bootstrap_prior.png")

def plot_bootstrap_recovery_weights(boots,hyp,weights,wMode,errorbar,filename):
    fig,ax = plt.subplots( figsize=(10,3.5))
    K,N = wMode.shape
    plt.xlim(0,N)
    plt.xlabel('Flash #',fontsize=12)
    plt.ylabel('Weight',fontsize=12)
    ax.tick_params(axis='both',labelsize=12)

    my_colors=['blue','green','purple','red']  
    for i in np.arange(0, K):
        plt.plot(wMode[i,:], "-", lw=3, color=my_colors[i])
        ax.fill_between(np.arange(len(wMode[i])), wMode[i,:]-2*errorbar[i], 
            wMode[i,:]+2*errorbar[i],facecolor=my_colors[i], alpha=0.1)    

        for boot in boots:
            plt.plot(boot[3][i,:], '--', color=my_colors[i], alpha=0.2)
    plt.tight_layout()
    if not (type(filename) == type(None)):
        plt.savefig(filename+"_bootstrap_weights.png")


def dropout_analysis(psydata, BIAS=True,TASK0=True, TASK1=False,TASKCR = False, OMISSIONS=False,OMISSIONS1=False, TIMING4=True,TIMING5=False):
    models =[]
    labels=[]
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Full-Task0')

    if BIAS:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=False, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Bias')
    if TASK0:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=False,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS,  OMISSIONS1=OMISSIONS1,TIMING4=TIMING4,TIMING5=TIMING5)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Task0')
    if TASK1:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=False, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Task1')
    if TASKCR:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=False, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('TaskCR')
    if (TASK0 & TASK1) | (TASK0 & TASKCR) | (TASK1 & TASKCR):
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=False,TASK1=False, TASKCR=False, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('All Task')
    if OMISSIONS:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=False, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Omissions')
    if OMISSIONS1:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=False,TIMING4=TIMING4,TIMING5=TIMING5)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Omissions1')
    if OMISSIONS & OMISSIONS1:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=False, OMISSIONS1=False,TIMING4=TIMING4,TIMING5=TIMING5)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('All Omissions')
    if TIMING4:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=False,TIMING5=TIMING5)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing4')
    if TIMING5:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=False)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing5')
    if TIMING4 & TIMING5:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=False,TIMING5=False)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('All timing')
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=False,TASK1=True, TASKCR=False, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Full-Task1')
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=True,TASK1=True, TASKCR=True, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Full-all Task')
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=True,TASK1=False, TASKCR=True, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Task 0/CR')
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=False, TASK0=True,TASK1=False, TASKCR=True, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING4=TIMING4,TIMING5=TIMING5)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Task 0/CR, no bias')
    return models,labels

def plot_dropout(models, labels,filename=None):
    plt.figure(figsize=(10,3.5))
    ax = plt.gca()
    for i in np.arange(0,len(models)):
        plt.plot(i, (1-models[i][1]/models[0][1])*100, 'ko')
    #plt.xlim(0,N)
    plt.xlabel('Model Component',fontsize=12)
    plt.ylabel('% change in evidence',fontsize=12)
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xticks(np.arange(0,len(models)))
    ax.set_xticklabels(labels)
    plt.tight_layout()
    ax.axhline(0,color='k',alpha=0.2)
    plt.ylim(ymax=5)
    if not (type(filename) == type(None)):
        plt.savefig(filename+"_dropout.png")

def plot_summaries(psydata):
    fig,ax = plt.subplots(nrows=8,ncols=1, figsize=(10,10),frameon=False)
    ax[0].plot(moving_mean(psydata['hits'],80),'b')
    ax[0].set_ylim(0,.15); ax[0].set_ylabel('hits')
    ax[1].plot(moving_mean(psydata['misses'],80),'r')
    ax[1].set_ylim(0,.15); ax[1].set_ylabel('misses')
    ax[2].plot(moving_mean(psydata['false_alarms'],80),'g')
    ax[2].set_ylim(0,.15); ax[2].set_ylabel('false_alarms')
    ax[3].plot(moving_mean(psydata['correct_reject'],80),'c')
    ax[3].set_ylim(0,.15); ax[3].set_ylabel('correct_reject')
    ax[4].plot(moving_mean(psydata['aborts'],80),'b')
    ax[4].set_ylim(0,.4); ax[4].set_ylabel('aborts')
    total_rate = moving_mean(psydata['hits'],80)+ moving_mean(psydata['misses'],80)+moving_mean(psydata['false_alarms'],80)+ moving_mean(psydata['correct_reject'],80)
    ax[5].plot(total_rate,'k')
    ax[5].set_ylim(0,.15); ax[5].set_ylabel('trial-rate')
    #ax[5].plot(total_rate,'b')
    ax[6].set_ylim(0,.15); ax[6].set_ylabel('d\' trials')
    ax[7].set_ylim(0,.15); ax[7].set_ylabel('d\' flashes')   
    for i in np.arange(0,len(ax)):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].yaxis.set_ticks_position('left')
        ax[i].xaxis.set_ticks_position('bottom')
        ax[i].set_xticklabels([])


def process_session(experiment_id):
    print("Pulling Data")
    session = get_data(experiment_id)
    print("Formating Data")
    psydata = format_session(session)
    filename = '/home/alex.piet/codebase/behavior/psy_fits/' + str(experiment_id) 
    print("Initial Fit")
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,TIMING4=True,OMISSIONS1=True)
    ypred,ypred_each = compute_ypred(psydata, wMode,weights)
    plot_weights(session,wMode, weights,psydata,errorbar=credibleInt, ypred = ypred,filename=filename)
    print("Bootstrapping")
    boots = bootstrap(10, psydata, ypred, weights, wMode)
    plot_bootstrap(boots, hyp, weights, wMode, credibleInt,filename=filename)
    print("Dropout Analysis")
    models, labels = dropout_analysis(psydata,TIMING5=True,OMISSIONS=True,OMISSIONS1=True)
    plot_dropout(models,labels,filename=filename)
    print("Cross Validation Analysis")
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    cv_pred = compute_cross_validation_ypred(psydata, cross_results,ypred)
    try:
        metadata = session.metadata
    except:
        metadata = []
    save(filename+".pkl", [models, labels, boots, hyp, evd, wMode, hess, credibleInt, weights, ypred,psydata,cross_results,cv_pred,metadata])
    plt.close('all')

def plot_session_summary_priors(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    alld = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass 
        else:
            sigmas = session_summary[0]
            weights = session_summary[1]
            ax.plot(np.arange(0,len(sigmas)),sigmas, 'o',alpha = 0.5)
            plt.yscale('log')
            plt.ylim(0.0001, 20)
            ax.set_xticks(np.arange(0,len(sigmas)))
            weights_list = []
            for i in sorted(weights.keys()):
                weights_list += [i]*weights[i]
            ax.set_xticklabels(weights_list,fontsize=12,rotation=90)
            plt.ylabel('Smoothing Prior, $\sigma$, smaller = smoother',fontsize=12)
            if type(alld) == type(None):
                alld = sigmas
            else:
                alld += sigmas
            counter +=1
    alld = alld/counter
    for i in np.arange(0, len(sigmas)):
        ax.plot([i-.25, i+.25],[alld[i],alld[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    ax.axhline(0.001,color='k',alpha=0.2)
    ax.axhline(0.01,color='k',alpha=0.2)
    ax.axhline(0.1,color='k',alpha=0.2)
    ax.axhline(1,color='k',alpha=0.2)
    ax.axhline(10,color='k',alpha=0.2)
    plt.yticks(fontsize=12)
    ax.xaxis.tick_top()
    ax.set_xlim(xmin=-.5)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"prior.png")

def plot_session_summary_dropout(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",cross_validation=True,savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(figsize=(7.2,6))
    alld = None
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory, cross_validation_dropout=cross_validation)
        except:
            pass 
        else:
            dropout = session_summary[2]
            labels  = session_summary[3]
            ax.plot(np.arange(0,len(dropout)),dropout, 'o',alpha=0.5)
            ax.set_xticks(np.arange(0,len(dropout)))
            ax.set_xticklabels(labels,fontsize=12, rotation = 90)
            plt.ylabel('% Change in Normalized Likelihood \n Smaller = Worse Fit',fontsize=12)

            if type(alld) == type(None):
                alld = dropout
            else:
                alld += dropout
            counter +=1
    alld = alld/counter
    plt.yticks(fontsize=12)
    for i in np.arange(0, len(dropout)):
        ax.plot([i-.25, i+.25],[alld[i],alld[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.xlim(-0.5,len(dropout) - 0.5)
    if savefig:
        if cross_validation:
            plt.savefig(directory+"summary_"+group_label+"dropout_cv.png")
        else:
            plt.savefig(directory+"summary_"+group_label+"dropout.png")

def plot_session_summary_weights(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/", savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    allW = None
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass 
        else:
            avgW = session_summary[4]
            weights  = session_summary[1]
            ax.plot(np.arange(0,len(avgW)),avgW, 'o',alpha=0.5)
            ax.set_xticks(np.arange(0,len(avgW)))
            plt.ylabel('Avg. Weights across each session',fontsize=12)

            if type(allW) == type(None):
                allW = avgW
            else:
                allW += avgW
            counter +=1
    allW = allW/counter
    for i in np.arange(0, len(avgW)):
        ax.plot([i-.25, i+.25],[allW[i],allW[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
    ax.set_xticklabels(weights_list,fontsize=12, rotation = 90)
    ax.xaxis.tick_top()
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.xlim(-0.5,len(avgW) - 0.5)
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weights.png")

def plot_session_summary_weight_range(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    allW = None
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass            
        else:
            rangeW = session_summary[5]
            weights  = session_summary[1]
            ax.plot(np.arange(0,len(rangeW)),rangeW, 'o',alpha=0.5)
            ax.set_xticks(np.arange(0,len(rangeW)))
            plt.ylabel('Range of Weights across each session',fontsize=12)

            if type(allW) == type(None):
                allW = rangeW
            else:
                allW += rangeW
            counter +=1
    allW = allW/counter
    for i in np.arange(0, len(rangeW)):
        ax.plot([i-.25, i+.25],[allW[i],allW[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
    ax.set_xticklabels(weights_list,fontsize=12, rotation = 90)
    ax.xaxis.tick_top()
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.xlim(-0.5,len(rangeW) - 0.5)
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_range.png")

def plot_session_summary_weight_scatter(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory= directory)
        except:
            pass 
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = []
            for i in sorted(weights.keys()):
                weights_list += [i]*weights[i]
            for i in np.arange(0,np.shape(W)[0]):
                if i < np.shape(W)[0]-1:
                    for j in np.arange(1, i+1):
                        ax[i,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                        ax[i,j-1].set_xticks([])
                        ax[i,j-1].set_yticks([])
                        for spine in ax[i,j-1].spines.values():
                            spine.set_visible(False)
                for j in np.arange(i+1,np.shape(W)[0]):
                    ax[i,j-1].axvline(0,color='k',alpha=0.05)
                    ax[i,j-1].axhline(0,color='k',alpha=0.05)
                    ax[i,j-1].plot(W[j,:], W[i,:],'o', alpha=0.01)
                    ax[i,j-1].set_xlabel(weights_list[j],fontsize=12)
                    ax[i,j-1].set_ylabel(weights_list[i],fontsize=12)
                    ax[i,j-1].xaxis.set_tick_params(labelsize=12)
                    ax[i,j-1].yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_scatter.png")

def plot_session_summary_dropout_scatter(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory, cross_validation_dropout=True)
        except:
            pass
        else:
            d = session_summary[2]
            l = session_summary[3]
            dropout = np.concatenate([d[1:3],[d[4]],[d[8]]])
            labels = l[1:3]+[l[4]]+[l[8]]
            for i in np.arange(0,np.shape(dropout)[0]):
                if i < np.shape(dropout)[0]-1:
                    for j in np.arange(1, i+1):
                        ax[i,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                        ax[i,j-1].set_xticks([])
                        ax[i,j-1].set_yticks([])
                        for spine in ax[i,j-1].spines.values():
                            spine.set_visible(False)
                for j in np.arange(i+1,np.shape(dropout)[0]):
                    ax[i,j-1].axvline(0,color='k',alpha=0.1)
                    ax[i,j-1].axhline(0,color='k',alpha=0.1)
                    ax[i,j-1].plot(dropout[j], dropout[i],'o',alpha=0.5)
                    ax[i,j-1].set_xlabel(labels[j],fontsize=12)
                    ax[i,j-1].set_ylabel(labels[i],fontsize=12)
                    ax[i,j-1].xaxis.set_tick_params(labelsize=12)
                    ax[i,j-1].yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"dropout_scatter.png")


def plot_session_summary_weight_avg_scatter(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = []
            for i in sorted(weights.keys()):
                weights_list += [i]*weights[i]
            for i in np.arange(0,np.shape(W)[0]):
                if i < np.shape(W)[0]-1:
                    for j in np.arange(1, i+1):
                        ax[i,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                        ax[i,j-1].set_xticks([])
                        ax[i,j-1].set_yticks([])
                        for spine in ax[i,j-1].spines.values():
                            spine.set_visible(False)
                for j in np.arange(i+1,np.shape(W)[0]):
                    ax[i,j-1].axvline(0,color='k',alpha=0.1)
                    ax[i,j-1].axhline(0,color='k',alpha=0.1)
                    meanWj = np.mean(W[j,:])
                    meanWi = np.mean(W[i,:])
                    stdWj = np.std(W[j,:])
                    stdWi = np.std(W[i,:])
                    ax[i,j-1].plot([meanWj, meanWj], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                    ax[i,j-1].plot(meanWj+[-stdWj,stdWj], [meanWi, meanWi],'k-',alpha=0.1)
                    ax[i,j-1].plot(meanWj, meanWi,'o',alpha=0.5)
                    ax[i,j-1].set_xlabel(weights_list[j],fontsize=12)
                    ax[i,j-1].set_ylabel(weights_list[i],fontsize=12)
                    ax[i,j-1].xaxis.set_tick_params(labelsize=12)
                    ax[i,j-1].yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_avg_scatter.png")

def plot_session_summary_weight_avg_scatter_task0(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
    allx = []
    ally = []
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = []
            for i in sorted(weights.keys()):
                weights_list += [i]*weights[i]
            xdex = np.where(np.array(weights_list) == 'task0')[0][0]
            ydex = np.where(np.array(weights_list) == 'omissions1')[0][0]
            ax.axvline(0,color='k',alpha=0.1)
            ax.axhline(0,color='k',alpha=0.1)
            meanWj = np.mean(W[xdex,:])
            meanWi = np.mean(W[ydex,:])
            allx.append(meanWj)
            ally.append(meanWi)
            stdWj = np.std(W[xdex,:])
            stdWi = np.std(W[ydex,:])
            ax.plot([meanWj, meanWj], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
            ax.plot(meanWj+[-stdWj,stdWj], [meanWi, meanWi],'k-',alpha=0.1)
            ax.plot(meanWj, meanWi,'o',alpha=0.5)
            ax.set_xlabel(weights_list[xdex],fontsize=12)
            ax.set_ylabel(weights_list[ydex],fontsize=12)
            ax.xaxis.set_tick_params(labelsize=12)
            ax.yaxis.set_tick_params(labelsize=12)
    x = np.array(allx).reshape((-1,1))
    y = np.array(ally)
    model = LinearRegression(fit_intercept=False).fit(x,y)
    sortx = np.sort(allx).reshape((-1,1))
    y_pred = model.predict(sortx)
    ax.plot(sortx,y_pred, 'r--')
    score = round(model.score(x,y),2)
    #plt.text(sortx[0]+.5,y_pred[0]-.5,"Omissions = "+str(round(model.coef_[0],2))+"*Task + " + str(round(model.intercept_,2))+"\nr^2 = "+str(score),color="r",fontsize=12)
    plt.text(sortx[0]+.5,y_pred[0]-.5,"Omissions = "+str(round(model.coef_[0],2))+"*Task \nr^2 = "+str(score),color="r",fontsize=12)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_avg_scatter_task0.png")
    return model



def plot_session_summary_weight_trajectory(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(nrows=4,ncols=1,figsize=(6,10))
    allW = None
    counter = 0
    xmax  =  []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = []
            for i in sorted(weights.keys()):
                weights_list += [i]*weights[i]
            for i in np.arange(0,np.shape(W)[0]):
                ax[i].plot(W[i,:],alpha = 0.2)
                ax[i].set_ylabel(weights_list[i],fontsize=12)

                xmax.append(len(W[i,:]))
                ax[i].set_xlim(0,np.max(xmax))
                ax[i].xaxis.set_tick_params(labelsize=12)
                ax[i].yaxis.set_tick_params(labelsize=12)
                if i == np.shape(W)[0] -1:
                    ax[i].set_xlabel('Flash #',fontsize=12)
            if type(allW) == type(None):
                allW = W[:,0:3900]
            else:
                allW += W[:,0:3900]
            counter +=1
    allW = allW/counter
    for i in np.arange(0,np.shape(W)[0]):
        ax[i].axhline(0, color='k')
        ax[i].plot(allW[i,:],'k',alpha = 1,lw=3)
        if i> 0:
            ax[i].set_ylim(ymin=-2.5)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_trajectory.png")

def get_cross_validation_dropout(cv_results):
    return np.sum([i['logli'] for i in cv_results]) 

def get_Excit_IDS(metadata):
    IDS =[]
    for m in metadata:
        if m['full_genotype'][0:5] == 'Slc17':
            IDS.append(m['ophys_experiment_id'])
    return IDS

def get_Inhib_IDS(metadata):
    IDS =[]
    for m in metadata:
        if not( m['full_genotype'][0:5] == 'Slc17'):
            IDS.append(m['ophys_experiment_id'])
    return IDS

def get_stage_names(IDS):
    stages = [[],[],[],[],[],[],[]]

    for id in IDS:
        print(id)
        try:    
            stage= get_stage(id)
        except:
            pass
        else:
            stages[int(stage[6])].append(id)
    return stages


def get_all_metadata(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/"):
    m = []
    for id in IDS:
        try:
            filename = directory + str(id) + ".pkl" 
            [models, labels, boots, hyp, evd, wMode, hess, credibleInt, weights, ypred,psydata,cross_results,cv_pred,metadata] = load(filename)
            m.append(metadata)
        except:
            pass
    
    return m
           
def get_session_summary(experiment_id,cross_validation_dropout=True,directory="/home/alex.piet/codebase/behavior/psy_fits/"):
    filename = directory + str(experiment_id) + ".pkl" 
    [models, labels, boots, hyp, evd, wMode, hess, credibleInt, weights, ypred,psydata,cross_results,cv_pred,metadata] = load(filename)
    # compute statistics
    dropout = []
    if cross_validation_dropout:
        for i in np.arange(0, len(models)):
            dropout.append(get_cross_validation_dropout(models[i][6]))
        dropout = np.array(dropout)
        dropout = (1-dropout/dropout[0])*100
    else:
        for i in np.arange(0, len(models)):
            dropout.append((1-models[i][1]/models[0][1])*100)
        dropout = np.array(dropout)
    avgW = np.mean(wMode,1)
    rangeW = np.ptp(wMode,1)
    return hyp['sigma'],weights,dropout,labels, avgW, rangeW,wMode

def plot_session_summary(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",savefig=False,group_label=""):
    plot_session_summary_priors(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_dropout(IDS,directory=directory,cross_validation=False,savefig=savefig,group_label=group_label)
    plot_session_summary_dropout(IDS,directory=directory,cross_validation=True,savefig=savefig,group_label=group_label)
    plot_session_summary_dropout_scatter(IDS, directory=directory, savefig=savefig, group_label=group_label)
    plot_session_summary_weights(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_range(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_scatter(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_avg_scatter(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_avg_scatter_task0(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_trajectory(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_logodds(IDS,directory=directory,savefig=savefig,group_label=group_label)


def compute_cross_validation(psydata, hyp, weights,folds=10):
    trainDs, testDs = Kfold_crossVal(psydata,F=folds)
    test_results = []
    for k in range(folds):
        print("running fold", k)
        _,_,wMode_K,_ = hyperOpt(trainDs[k], hyp, weights, ['sigma'])
        logli, gw = Kfold_crossVal_check(testDs[k], wMode_K, trainDs[k]['missing_trials'], weights)
        res = {'logli' : np.sum(logli), 'gw' : gw, 'test_inds' : testDs[k]['test_inds']}
        test_results += [res]
    return test_results

def compute_cross_validation_ypred(psydata,test_results,ypred):
    # combine each folds predictions
    myrange = np.arange(0, len(psydata['y']))
    xval_mask = np.ones(len(myrange)).astype(bool)
    X = np.array([i['gw'] for i in test_results]).flatten()
    test_inds = np.array([i['test_inds'] for i in test_results]).flatten()
    inrange = np.where((test_inds >= 0) & (test_inds < len(psydata['y'])))[0]
    inds = [i for i in np.argsort(test_inds) if i in inrange]
    X = X[inds]
    # because length of trial might not be perfectly divisible, there are untested indicies
    untested_inds = [j for j in myrange if j not in test_inds]
    untested_inds = [np.where(myrange == i)[0][0] for i in untested_inds]
    xval_mask[untested_inds] = False
    cv_pred = 1/(1+np.exp(-X))
    # Fill in untested indicies with ypred
    full_pred = copy.copy(ypred)
    full_pred[np.where(xval_mask==True)[0]] = cv_pred
    return  full_pred


def plot_session_summary_logodds(IDS,directory="/home/alex.piet/codebase/behavior/psy_fits/",savefig=False,group_label=""):
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4.5))
    logodds=[]
    for id in IDS:
        try:
            #session_summary = get_session_summary(id)
            filenamed = directory + str(id) + ".pkl" 
            [models, labels, boots, hyp, evd, wMode, hess, credibleInt, weights, ypred,psydata,cross_results,cv_pred,metadata] = load(filenamed)
        except:
            pass
        else:
            lickedp = np.mean(ypred[psydata['y'] ==2])
            nolickp = np.mean(ypred[psydata['y'] ==1])
            ax[0].plot(nolickp,lickedp, 'o', alpha = 0.5)
            logodds.append(np.log(lickedp/nolickp))
    ax[0].set_ylabel('P(lick|lick)', fontsize=12)
    ax[0].set_xlabel('P(lick|no-lick)', fontsize=12)
    ax[0].plot([0,1],[0,1], 'k--',alpha=0.2)
    ax[0].xaxis.set_tick_params(labelsize=12)
    ax[0].yaxis.set_tick_params(labelsize=12)
    ax[0].set_ylim(0,1)
    ax[0].set_xlim(0,1)
    ax[1].hist(np.array(logodds),bins=30)
    ax[1].set_ylabel('Count', fontsize=12)
    ax[1].set_xlabel('Log-Odds', fontsize=12)
    ax[1].xaxis.set_tick_params(labelsize=12)
    ax[1].yaxis.set_tick_params(labelsize=12)

    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_logodds.png")


def get_all_weights(IDS,directory='/home/alex.piet/codebase/behavior/psy_fits/'):
    weights = None
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass
        else:
            if weights is None:
                weights = session_summary[6]
            else:
                weights = np.concatenate([weights, session_summary[6]],1)
    return weights

def load_fit(ID, directory='/home/alex.piet/codebase/behavior/psy_fits/'):
    filename = directory + str(ID) + ".pkl" 
    output = load(filename)
    labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
    fit = dict((x,y) for x,y in zip(labels, output))
    fit['ID'] = ID
    if os.path.isfile(directory+str(ID) + "_clusters.pkl"):
        clusters = load(directory+str(ID) + "_clusters.pkl")
        fit['clusters'] = clusters
    else:
        fit = cluster_fit(fit,directory=directory)
    return fit

def plot_fit(ID, cluster_labels=None,fit=None, session=None, directory='/home/alex.piet/codebase/behavior/psy_fits/'):
    if not (type(fit) == type(dict())):
        fit = load_fit(ID, directory=directory)
    if type(session) == type(None):
        session = get_data(ID)
    plot_weights(session,fit['wMode'], fit['weights'],fit['psydata'],errorbar=fit['credibleInt'], ypred = fit['ypred'],cluster_labels=cluster_labels)
    return fit, session
   
def cluster_fit(fit,directory='/home/alex.piet/codebase/behavior/psy_fits/'):
    numc= range(2,5)
    cluster = dict()
    for i in numc:
        output = cluster_weights(fit['wMode'],i)
        cluster[str(i)] = output
    fit['cluster'] = cluster
    filename = directory + str(fit['ID']) + "_clusters.pkl" 
    save(filename, cluster) 
    return fit

def cluster_weights(wMode,num_clusters):
    output = k_means(transform(wMode.T),num_clusters)
    return output

def check_clustering(wMode,numC=5):
    fig,ax = plt.subplots(nrows=numC,ncols=1)
    scores = []
    for j in range(0,numC):
        for i in range(0,4):
            ax[j].plot(transform(wMode[i,:]))
        output = k_means(transform(wMode.T),j+1)
        cp = np.where(~(np.diff(output[1]) == 0))[0]
        cp = np.concatenate([[0], cp, [len(output[1])]])
        colors = ['r','b','g','c','m','k','y']
        for i in range(0, len(cp)-1):
            ax[j].axvspan(cp[i],cp[i+1],color=colors[output[1][cp[i]+1]], alpha=0.1)
        ax[j].set_ylim(0,1)
        ax[j].set_xlim(0,len(wMode[0,:]))
        ax[j].set_ylabel(str(j+2)+" clusters")
        ax[j].set_xlabel('Flash #')
        scores.append(output[2])
    return scores

def check_all_clusters(IDS, numC=8):
    all_scores = []
    for i in IDS:
        scores = []
        try:
            wMode = get_all_weights([i])
        except:
            pass
        else:
            if not (type(wMode) == type(None)):
                for j in range(0,numC):
                    output = k_means(transform(wMode.T),j+1)
                    scores.append(output[2])
                all_scores.append(scores)
    
    plt.figure()
    for i in np.arange(0,len(all_scores)):
        plt.plot(np.arange(1,j+2), all_scores[i]/all_scores[i][0],'k-',alpha=0.3)    
    plt.ylabel('Normalized error')
    plt.xlabel('number of clusters')
    
    
