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
from allensdk.brain_observatory.behavior import behavior_ophys_session as bos
from allensdk.brain_observatory.behavior import stimulus_processing
from allensdk.internal.api import behavior_lims_api as bla
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.internal.api import behavior_ophys_api as boa
from sklearn.linear_model import LinearRegression
from sklearn.cluster import k_means
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import pandas as pd
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc
#from allensdk.brain_observatory.behavior import behavior_project_cache as bpc
from sklearn.decomposition import PCA
import seaborn as sns
from functools import reduce
import psy_timing_tools as pt
import psy_metrics_tools as pm

INTERNAL= True
global_directory="/home/alex.piet/codebase/behavior/psy_fits_v3/"

def load(filepath):
    '''
        Handy function for loading a pickle file. 
    '''
    filetemp = open(filepath,'rb')
    data    = pickle.load(filetemp)
    filetemp.close()
    return data

def save(filepath, variables):
    '''
        Handy function for saving variables to a pickle file. 
    '''
    file_temp = open(filepath,'wb')
    pickle.dump(variables, file_temp)
    file_temp.close()

def get_data(experiment_id,stage="",load_dir = r'/allen/aibs/technology/nicholasc/behavior_ophys'):
    '''
        Loads data from SDK interface
        ARGS: experiment_id to load
    '''
    if not INTERNAL:
        experiment_id = get_session_id(experiment_id)
    cache = get_cache()
    session = cache.get_session(experiment_id)
    return session

def get_stage(experiment_id):
    api=boa.BehaviorOphysLimsApi(experiment_id)
    return api.get_task_parameters()['stage']

def check_grace_windows(session,time_point):
    '''
        v1 code
        Returns true if the time point is inside the grace period after reward delivery from an earned reward or auto-reward
    '''
    hit_end_times = session.trials.stop_time[session.trials.hit].values
    hit_response_time = session.trials.response_latency[session.trials.hit].values + session.trials.change_time[session.trials.hit].values
    inside_grace_window = np.any((hit_response_time < time_point ) & (hit_end_times > time_point))
    
    auto_reward_time = session.trials.change_time[(session.trials.auto_rewarded) & (~session.trials.aborted)] + .5
    auto_end_time = session.trials.stop_time[(session.trials.auto_rewarded) & (~session.trials.aborted)]
    inside_auto_window = np.any((auto_reward_time < time_point) & (auto_end_time > time_point))
    return inside_grace_window | inside_auto_window

def annotate_stimulus_presentations(session):
    '''
        Adds columns to the stimulus_presentation table describing whether certain task events happened during that flash
        Inputs:
        session, the SDK session object
    
        Appends columns:
        licked, True if the mouse licked during this flash, does not care about the response window
        hits,   True if the mouse licked on a change flash. 
        misses, True if the mouse did not lick on a change flash
        aborts, True if the mouse licked on a non-change-flash. THIS IS NOT THE SAME AS THE TRIALS TABLE ABORT DEFINITION.
                licks on sequential flashes that are during the abort time out period are counted as aborts here.
                this abort list should only be used for simple visualization purposes
        in_grace_period, True if this flash occurs during the 0.75 - 4.5 period after the onset of a hit change
        false_alarm,    True if the mouse licked on a sham-change-flash
        correct_reject, True if the mouse did not lick on a sham-change-flash
        auto_rewards,   True if there was an auto-reward during this flash
    '''
    session.stimulus_presentations['licked'] = ~session.stimulus_presentations.licks.str[0].isnull()
    session.stimulus_presentations['hits'] = session.stimulus_presentations['licked'] & session.stimulus_presentations['change']
    session.stimulus_presentations['misses'] = ~session.stimulus_presentations['licked'] & session.stimulus_presentations['change']
    session.stimulus_presentations['aborts'] = session.stimulus_presentations['licked'] & ~session.stimulus_presentations['change']
    session.stimulus_presentations['in_grace_period'] = (session.stimulus_presentations['time_from_last_change'] <= 4.5) & (session.stimulus_presentations['time_from_last_reward'] <=4.5)
    session.stimulus_presentations.at[session.stimulus_presentations['in_grace_period'],'aborts'] = False # Remove Aborts that happened during grace period
    session.stimulus_presentations['false_alarm'] = False
    session.stimulus_presentations['correct_reject'] = False
    session.stimulus_presentations['auto_rewards'] = False

    # These ones require iterating the fucking trials table, and is super slow
    for i in session.stimulus_presentations.index:
        found_it=True
        trial = session.trials[(session.trials.start_time <= session.stimulus_presentations.at[i,'start_time']) & (session.trials.stop_time >=session.stimulus_presentations.at[i,'start_time'] + 0.25)]
        if len(trial) > 1:
            raise Exception("Could not isolate a trial for this flash")
        if len(trial) == 0:
            trial = session.trials[(session.trials.start_time <= session.stimulus_presentations.at[i,'start_time']) & (session.trials.stop_time+0.75 >= session.stimulus_presentations.at[i,'start_time'] + 0.25)]  
            if ( len(trial) == 0 ) & (session.stimulus_presentations.at[i,'start_time'] > session.trials.start_time.values[-1]):
                trial = session.trials[session.trials.index == session.trials.index[-1]]
            elif np.sum(session.trials.aborted) == 0:
                found_it=False
            elif len(trial) == 0:
                raise Exception("Could not find a trial for this flash")
        if found_it:
            if trial['false_alarm'].values[0]:
                if (trial.change_time.values[0] >= session.stimulus_presentations.at[i,'start_time']) & (trial.change_time.values[0] <= session.stimulus_presentations.at[i,'stop_time'] ):
                    session.stimulus_presentations.at[i,'false_alarm'] = True
            if trial['correct_reject'].values[0]:
                if (trial.change_time.values[0] >= session.stimulus_presentations.at[i,'start_time']) & (trial.change_time.values[0] <= session.stimulus_presentations.at[i,'stop_time'] ):
                    session.stimulus_presentations.at[i,'correct_reject'] = True
            if trial['auto_rewarded'].values[0]:
                if (trial.change_time.values[0] >= session.stimulus_presentations.at[i,'start_time']) & (trial.change_time.values[0] <= session.stimulus_presentations.at[i,'stop_time'] ):
                    session.stimulus_presentations.at[i,'auto_rewards'] = True


def format_session(session,format_options):
    '''
        Formats the data into the requirements of Psytrack
        ARGS:
            data outputed from SDK
            format_options, a dictionary with keys:
                fit_bouts, if True (Default), then fits to the start of each licking bout, instead of each lick
                timing0/1, if True (Default), then timing is a vector of 0s and 1s, otherwise, is -1/+1
                mean_center, if True, then regressors are mean-centered
        Returns:
            data formated for psytrack. A dictionary with key/values:
            psydata['y'] = a vector of no-licks (1) and licks(2) for each flashes
            psydata['inputs'] = a dictionary with each key an input ('random','timing', 'task', etc)
                each value has a 2D array of shape (N,M), where N is number of flashes, and M is 1 unless you want to look at history/flash interaction terms
    '''     
    if len(session.licks) < 10:
        raise Exception('Less than 10 licks in this session')   

    defaults = {'fit_bouts':True,'timing0/1':True,'mean_center':False}
    for k in defaults.keys():
        if k not in format_options:
            format_options[k] = defaults[k]

    # Build Dataframe of flashes
    annotate_stimulus_presentations(session)
    df = pd.DataFrame(data = session.stimulus_presentations.start_time)
    if format_options['fit_bouts']:
        licks = session.stimulus_presentations.bout_start.values
        df['y'] = np.array([2 if x else 1 for x in licks])
    else:
        licks = session.stimulus_presentations.licks.str[0].isnull()
        df['y'] = np.array([1 if x else 2 for x in licks])
    df['hits'] = session.stimulus_presentations.hits
    df['misses'] = session.stimulus_presentations.misses
    df['false_alarm'] = session.stimulus_presentations.false_alarm
    df['correct_reject'] = session.stimulus_presentations.correct_reject
    df['aborts'] = session.stimulus_presentations.aborts
    df['auto_rewards'] = session.stimulus_presentations.auto_rewards
    df['start_time'] = session.stimulus_presentations.start_time
    df['change'] = session.stimulus_presentations.change
    df['omitted'] = session.stimulus_presentations.omitted  
    df['licked'] = session.stimulus_presentations.licked
    df['included'] = True
 
    # Build Dataframe of regressors
    if format_options['fit_bouts']:
        df['bout_start'] = session.stimulus_presentations['bout_start']
        df['bout_end'] = session.stimulus_presentations['bout_end']
        df['flashes_since_last_lick'] = session.stimulus_presentations.groupby(session.stimulus_presentations['bout_end'].cumsum()).cumcount(ascending=True)
        df['in_bout'] = session.stimulus_presentations['bout_start'].cumsum() > session.stimulus_presentations['bout_end'].cumsum()
        df['in_bout'] = np.array([1 if x else 0 for x in df['in_bout'].shift(fill_value=False)])
        df['task0'] = np.array([1 if x else 0 for x in df['change']])
        df['task1'] = np.array([1 if x else -1 for x in df['change']])
        df['taskCR'] = np.array([0 if x else -1 for x in df['change']])
        df['omissions'] = np.array([1 if x else 0 for x in df['omitted']])
        df['omissions1'] = np.array([x for x in np.concatenate([[0], df['omissions'].values[0:-1]])])
        if format_options['timing0/1']:
            df['timing1'] =  np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==0])
            df['timing2'] =  np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==1])
            df['timing3'] =  np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==2])
            df['timing4'] =  np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==3])
            df['timing5'] =  np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==4])
            df['timing6'] =  np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==5])
            df['timing7'] =  np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==6])
            df['timing8'] =  np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==7])
            df['timing9'] =  np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==8])
            df['timing10'] = np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() ==9])
        else:
            df['timing1'] =  np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==0])
            df['timing2'] =  np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==1])
            df['timing3'] =  np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==2])
            df['timing4'] =  np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==3])
            df['timing5'] =  np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==4])
            df['timing6'] =  np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==5])
            df['timing7'] =  np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==6])
            df['timing8'] =  np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==7])
            df['timing9'] =  np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==8])
            df['timing10'] = np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() ==9])
        full_df = copy.copy(df)
        df = df[df['in_bout']==0] 
        df['missing_trials'] = np.concatenate([np.diff(df.index)-1,[0]])
    else:
        df['task0'] = np.array([1 if x else 0 for x in df['change']])
        df['task1'] = np.array([1 if x else -1 for x in df['change']])
        df['taskCR'] = np.array([0 if x else -1 for x in df['change']])
        df['omissions'] = np.array([1 if x else 0 for x in df['omitted']])
        df['omissions1'] = np.concatenate([[0], df['omissions'].values[0:-1]])
        df['flashes_since_last_lick'] = df.groupby(df['licked'].cumsum()).cumcount(ascending=True)
        if format_options['timing0/1']:
            df['timing2'] = np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() >=2])
            df['timing3'] = np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() >=3])
            df['timing4'] = np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() >=4])
            df['timing5'] = np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() >=5])
            df['timing6'] = np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() >=6])
            df['timing7'] = np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() >=7])
            df['timing8'] = np.array([1 if x else 0 for x in df['flashes_since_last_lick'].shift() >=8]) 
        else:
            df['timing2'] = np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() >=2])
            df['timing3'] = np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() >=3])
            df['timing4'] = np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() >=4])
            df['timing5'] = np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() >=5])
            df['timing6'] = np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() >=6])
            df['timing7'] = np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() >=7])
            df['timing8'] = np.array([1 if x else -1 for x in df['flashes_since_last_lick'].shift() >=8])
        df['missing_trials'] = np.array([ 0 for x in df['change']])
        full_df = copy.copy(df)

    # Package into dictionary for psytrack
    inputDict ={'task0': df['task0'].values[:,np.newaxis],
                'task1': df['task1'].values[:,np.newaxis],
                'taskCR': df['taskCR'].values[:,np.newaxis],
                'omissions' : df['omissions'].values[:,np.newaxis],
                'omissions1' : df['omissions1'].values[:,np.newaxis],
                'timing1': df['timing1'].values[:,np.newaxis],
                'timing2': df['timing2'].values[:,np.newaxis],
                'timing3': df['timing3'].values[:,np.newaxis],
                'timing4': df['timing4'].values[:,np.newaxis],
                'timing5': df['timing5'].values[:,np.newaxis],
                'timing6': df['timing6'].values[:,np.newaxis],
                'timing7': df['timing7'].values[:,np.newaxis],
                'timing8': df['timing8'].values[:,np.newaxis],
                'timing9': df['timing9'].values[:,np.newaxis],
                'timing10': df['timing10'].values[:,np.newaxis],
                'missing_trials': df['missing_trials'].values[:,np.newaxis] }
    psydata = { 'y': df['y'].values, 
                'inputs':inputDict, 
                'false_alarms': df['false_alarm'].values,
                'correct_reject': df['correct_reject'].values,
                'hits': df['hits'].values,
                'misses':df['misses'].values,
                'aborts':df['aborts'].values,
                'auto_rewards':df['auto_rewards'].values,
                'start_times':df['start_time'].values,
                'flash_ids': df.index.values,
                'df':df,
                'full_df':full_df }
    try: 
        psydata['session_label'] = [session.metadata['stage']]
    except:
        psydata['session_label'] = ['Unknown Label']  
    return psydata


def get_trial(session, start_time,stop_time):
    ''' 
        v1 code
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
    
def fit_weights(psydata, BIAS=True,TASK0=True, TASK1=False,TASKCR = False, OMISSIONS=False,OMISSIONS1=True,TIMING1=True,TIMING2=True,TIMING3=True, TIMING4=True,TIMING5=True,TIMING6=True,TIMING7=True,TIMING8=True,TIMING9=True,TIMING10=True,fit_overnight=False):
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
    if TIMING1: weights['timing1'] = 1
    if TIMING2: weights['timing2'] = 1
    if TIMING3: weights['timing3'] = 1
    if TIMING4: weights['timing4'] = 1
    if TIMING5: weights['timing5'] = 1
    if TIMING6: weights['timing6'] = 1
    if TIMING7: weights['timing7'] = 1
    if TIMING8: weights['timing8'] = 1
    if TIMING9: weights['timing9'] = 1
    if TIMING10: weights['timing10'] = 1
    print(weights)

    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
            'sigma':[2**-4.]*K,
            'sigDay': 2**4}
    if fit_overnight:
        optList=['sigma','sigDay']
    else:
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

def inverse_transform(series):
    return -np.log((1/series) - 1)

def transform(series):
    '''
        passes the series through the logistic function
    '''
    return 1/(1+np.exp(-(series)))

def get_flash_index_session(session, time_point):
    '''
        v1 code
        Returns the flash index of a time point
    '''
    return np.where(session.stimulus_presentations.start_time.values < time_point)[0][-1]

def get_flash_index(psydata, time_point):
    '''
        v1 code
        Returns the flash index of a time point
    '''
    if time_point > psydata['start_times'][-1] + 0.75:
        return np.nan
    return np.where(np.array(psydata['start_times']) < time_point)[0][-1]


def moving_mean(values, window):
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm

def plot_weights(wMode,weights,psydata,errorbar=None, ypred=None,START=0, END=0,remove_consumption=True,validation=True,session_labels=None, seedW = None,ypred_each = None,filename=None,cluster_labels=None,smoothing_size=50,num_clusters=None):
    '''
        Plots the fit results by plotting the weights in linear and probability space. 
    
    '''
    K,N = wMode.shape    
    if START <0: START = 0
    if START > N: raise Exception(" START > N")
    if END <=0: END = N
    if END > N: END = N
    if START >= END: raise Exception("START >= END")

    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
   
    #my_colors=['blue','green','purple','red','coral','pink','yellow','cyan','dodgerblue','peru','black','grey','violet']  
    my_colors = sns.color_palette("hls",len(weights.keys()))
    if 'dayLength' in psydata:
        dayLength = np.concatenate([[0],np.cumsum(psydata['dayLength'])])
    else:
        dayLength = []

    cluster_ax = 3
    if (not (type(ypred) == type(None))) & validation:
        fig,ax = plt.subplots(nrows=4,ncols=1, figsize=(10,10))
        #ax[3].plot(ypred, 'k',alpha=0.3,label='Full Model')
        ax[3].plot(moving_mean(ypred,smoothing_size), 'k',alpha=0.3,label='Full Model (n='+str(smoothing_size)+ ')')
        if not( type(ypred_each) == type(None)):
            for i in np.arange(0, len(weights_list)):
                ax[3].plot(ypred_each[:,i], linestyle="-", lw=3, alpha = 0.3,color=my_colors[i],label=weights_list[i])        
        ax[3].plot(moving_mean(psydata['y']-1,smoothing_size), 'b',alpha=0.5,label='data (n='+str(smoothing_size)+ ')')
        ax[3].set_ylim(0,1)
        ax[3].set_ylabel('Lick Prob',fontsize=12)
        ax[3].set_xlabel('Flash #',fontsize=12)
        ax[3].set_xlim(START,END)
        ax[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
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
        #cluster_colors = ['r','b','g','c','m','k','y']
        if type(num_clusters) == type(None):
            num_clusters = len(np.unique(cluster_labels))
        cluster_colors = sns.color_palette("hls",num_clusters)
        for i in range(0, len(cp)-1):
            ax[cluster_ax].axvspan(cp[i],cp[i+1],color=cluster_colors[cluster_labels[cp[i]+1]], alpha=0.3)
    for i in np.arange(0, len(weights_list)):
        ax[0].plot(wMode[i,:], linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])        
        ax[0].fill_between(np.arange(len(wMode[i])), wMode[i,:]-2*errorbar[i], 
            wMode[i,:]+2*errorbar[i],facecolor=my_colors[i], alpha=0.1)    
        ax[1].plot(transform(wMode[i,:]), linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])
        ax[1].fill_between(np.arange(len(wMode[i])), transform(wMode[i,:]-2*errorbar[i]), 
            transform(wMode[i,:]+2*errorbar[i]),facecolor=my_colors[i], alpha=0.1)                  
        if not (type(seedW) == type(None)):
            ax[0].plot(seedW[i,:], linestyle="--", lw=2, color=my_colors[i], label= "seed "+weights_list[i])
            ax[1].plot(transform(seedW[i,:]), linestyle="--", lw=2, color=my_colors[i], label= "seed "+weights_list[i])
    ax[0].plot([0,np.shape(wMode)[1]], [0,0], 'k--',alpha=0.2)
    ax[0].set_ylabel('Weight',fontsize=12)
    ax[0].set_xlabel('Flash #',fontsize=12)
    ax[0].set_xlim(START,END)
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].tick_params(axis='both',labelsize=12)
    for i in np.arange(0, len(dayLength)-1):
        ax[0].axvline(dayLength[i],color='k',alpha=0.2)
        if not type(session_labels) == type(None):
            ax[0].text(dayLength[i],ax[0].get_ylim()[1], session_labels[i],rotation=25)
    ax[1].set_ylim(0,1)
    ax[1].set_ylabel('Lick Prob',fontsize=12)
    ax[1].set_xlabel('Flash #',fontsize=12)
    ax[1].set_xlim(START,END)
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].tick_params(axis='both',labelsize=12)
    for i in np.arange(0, len(dayLength)-1):
        ax[1].plot([dayLength[i], dayLength[i]],[0,1], 'k-',alpha=0.2)

    if validation:
        #first_start = session.trials.loc[0].start_time
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
    '''
        Debugging function that plots the licks in psydata against the session objects
    '''
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
    """
    v1 code
    Simulates weights, in addition to inputs and multiple realizations
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
    '''
        Samples the model. This function is a bit broken because it uses the original licking times to determine the timing strategies, and not the new licks that have been sampled. But it works fairly well
    '''
    bootdata = copy.copy(psydata)    
    if not ('ypred' in bootdata):
        raise Exception('You need to compute y-prediction first')
    temp = np.random.random(np.shape(bootdata['ypred'])) < bootdata['ypred']
    licks = np.array([2 if x else 1 for x in temp])   
    bootdata['y'] = licks
    return bootdata


def bootstrap_model(psydata, ypred, weights,seedW,plot_this=True):
    '''
        Does one bootstrap of the data and model prediction
    '''
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
        plot_weights(bwMode, weights, bootdata, errorbar=bcredibleInt, validation=False,seedW =seedW )
    return (bootdata, bhyp, bevd, bwMode, bhess, bcredibleInt)

def bootstrap(numboots, psydata, ypred, weights, seedW, plot_each=False):
    '''
    Performs a bootstrapping procedure on a fit by sampling the model repeatedly and then fitting the samples 
    '''
    boots = []
    for i in np.arange(0,numboots):
        print(i)
        boot = bootstrap_model(psydata, ypred, weights, seedW,plot_this=plot_each)
        boots.append(boot)
    return boots

def plot_bootstrap(boots, hyp, weights, seedW, credibleInt,filename=None):
    '''
        Calls each of the plotting functions for the weights and the prior
    '''
    plot_bootstrap_recovery_prior(boots,hyp, weights,filename)
    plot_bootstrap_recovery_weights(boots,hyp, weights,seedW,credibleInt,filename)


def plot_bootstrap_recovery_prior(boots,hyp,weights,filename):
    '''
        Plots how well the bootstrapping procedure recovers the hyper-parameter priors. Plots the seed prior and each bootstrapped value
    '''
    fig,ax = plt.subplots(figsize=(3,4))
    #my_colors=['blue','green','purple','red','coral','pink','yellow','cyan','dodgerblue','peru','black','grey','violet']
    my_colors = sns.color_palette("hls",len(weights.keys()))
    plt.yscale('log')
    plt.ylim(0.001, 20)
    ax.set_xticks(np.arange(0,len(hyp['sigma'])))
    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
    ax.set_xticklabels(weights_list,rotation=90)
    plt.ylabel('Smoothing Prior, $\sigma$ \n <-- More Smooth      More Variable -->')
    for boot in boots:
        plt.plot(boot[1]['sigma'], 'kx',alpha=0.5)
    for i in np.arange(0, len(hyp['sigma'])):
        plt.plot(i,hyp['sigma'][i], 'o', color=my_colors[i])

    plt.tight_layout()
    if not (type(filename) == type(None)):
        plt.savefig(filename+"_bootstrap_prior.png")

def plot_bootstrap_recovery_weights(boots,hyp,weights,wMode,errorbar,filename):
    '''
        plots the output of bootstrapping on the weight trajectories, plots the seed values and each bootstrapped recovered value   
    '''
    fig,ax = plt.subplots( figsize=(10,3.5))
    K,N = wMode.shape
    plt.xlim(0,N)
    plt.xlabel('Flash #',fontsize=12)
    plt.ylabel('Weight',fontsize=12)
    ax.tick_params(axis='both',labelsize=12)

    #my_colors=['blue','green','purple','red','coral','pink','yellow','cyan','dodgerblue','peru','black','grey','violet']
    my_colors = sns.color_palette("hls",len(weights.keys()))
    for i in np.arange(0, K):
        plt.plot(wMode[i,:], "-", lw=3, color=my_colors[i])
        ax.fill_between(np.arange(len(wMode[i])), wMode[i,:]-2*errorbar[i], 
            wMode[i,:]+2*errorbar[i],facecolor=my_colors[i], alpha=0.1)    

        for boot in boots:
            plt.plot(boot[3][i,:], '--', color=my_colors[i], alpha=0.2)
    plt.tight_layout()
    if not (type(filename) == type(None)):
        plt.savefig(filename+"_bootstrap_weights.png")


def dropout_analysis(psydata, BIAS=True,TASK0=True, TASK1=False,TASKCR = False, OMISSIONS=True,OMISSIONS1=True,TIMING1=True, TIMING2=True,TIMING3=True, TIMING4=True,TIMING5=True,TIMING6=True,TIMING7=True,TIMING8=True,TIMING9=True, TIMING10=True):
    '''
        Computes a dropout analysis for the data in psydata. In general, computes a full set, and then removes each feature one by one. Also computes hard-coded combinations of features
        Returns a list of models and a list of labels for each dropout
    '''
    models =[]
    labels=[]
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1, TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Full-Task0')

    if BIAS:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=False, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Bias')
    if TASK0:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=False,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS,  OMISSIONS1=OMISSIONS1, TIMING1=TIMING1, TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Task0')
    if TASK1:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=False, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1,  TIMING1=TIMING1, TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Task1')
    if TASKCR:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=False, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('TaskCR')
    if (TASK0 & TASK1) | (TASK0 & TASKCR) | (TASK1 & TASKCR):
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=False,TASK1=False, TASKCR=False, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('All Task')
    if OMISSIONS:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=False, OMISSIONS1=OMISSIONS1,  TIMING1=TIMING1, TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Omissions')
    if OMISSIONS1:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=False, TIMING1=TIMING1, TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Omissions1')
    if OMISSIONS & OMISSIONS1:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=False, OMISSIONS1=False, TIMING1=TIMING1, TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('All Omissions')
    if TIMING1:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=False,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing1')
    if TIMING2:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=False,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing2')
    if TIMING3:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=False,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing3')
    if TIMING4:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=False,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing4')
    if TIMING5:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=False,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing5')
    if TIMING6:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1,  TIMING1=TIMING1, TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=False,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing6')
    if TIMING7:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=False,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing7')
    if TIMING8:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=False,TIMING9=TIMING9,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing8')
    if TIMING9:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=False,TIMING10=TIMING10)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing9')
    if TIMING10:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=False)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing10')

    if TIMING1 & TIMING2:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=False,  TIMING2=False,TIMING3=True,TIMING4=True,TIMING5=True,TIMING6=True,TIMING7=True,TIMING8=True,TIMING9=True,TIMING10=True)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing1/2')
    if TIMING3 & TIMING4:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=True,  TIMING2=True,TIMING3=False,TIMING4=False,TIMING5=True,TIMING6=True,TIMING7=True,TIMING8=True,TIMING9=True,TIMING10=True)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing3/4')
    if TIMING5 & TIMING6:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=True,  TIMING2=True,TIMING3=True,TIMING4=True,TIMING5=False,TIMING6=False,TIMING7=True,TIMING8=True,TIMING9=True,TIMING10=True)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing5/6')
    if TIMING7 & TIMING8:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=True,  TIMING2=True,TIMING3=True,TIMING4=True,TIMING5=True,TIMING6=True,TIMING7=False,TIMING8=False,TIMING9=True,TIMING10=True)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing7/8')
    if TIMING9 & TIMING10:
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=True,  TIMING2=True,TIMING3=True,TIMING4=True,TIMING5=True,TIMING6=True,TIMING7=True,TIMING8=True,TIMING9=False,TIMING10=False)    
        cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
        models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
        labels.append('Timing9/10')

    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=False,  TIMING2=False,TIMING3=False,TIMING4=False,TIMING5=False,TIMING6=True,TIMING7=True,TIMING8=True,TIMING9=True,TIMING10=True)    
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Timing 1-5')

    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=True,  TIMING2=True,TIMING3=True,TIMING4=True,TIMING5=True,TIMING6=False,TIMING7=False,TIMING8=False,TIMING9=False,TIMING10=False)    
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Timing 6-10')

    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=TASK0,TASK1=TASK1, TASKCR=TASKCR, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=False,  TIMING2=False,TIMING3=False,TIMING4=False,TIMING5=False,TIMING6=False,TIMING7=False,TIMING8=False,TIMING9=False,TIMING10=False)    
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('All timing')

    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=False,TASK1=True, TASKCR=False, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Full-Task1')
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=True,TASK1=True, TASKCR=True, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Full-all Task')
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=BIAS, TASK0=True,TASK1=False, TASKCR=True, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Task 0/CR')
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,BIAS=False, TASK0=True,TASK1=False, TASKCR=True, OMISSIONS=OMISSIONS, OMISSIONS1=OMISSIONS1, TIMING1=TIMING1,  TIMING2=TIMING2,TIMING3=TIMING3,TIMING4=TIMING4,TIMING5=TIMING5,TIMING6=TIMING6,TIMING7=TIMING7,TIMING8=TIMING8,TIMING9=TIMING9,TIMING10=TIMING10)
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    models.append((hyp, evd, wMode, hess, credibleInt,weights,cross_results))
    labels.append('Task 0/CR, no bias')

    return models,labels

def plot_dropout(models, labels,filename=None):
    '''
        Plots the dropout results for a single session
        
    '''
    plt.figure(figsize=(10,3.5))
    ax = plt.gca()
    for i in np.arange(0,len(models)):
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
        plt.plot(i, (1-models[i][1]/models[0][1])*100, 'ko')
    #plt.xlim(0,N)
    plt.xlabel('Model Component',fontsize=12)
    plt.ylabel('% change in evidence',fontsize=12)
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xticks(np.arange(0,len(models)))
    ax.set_xticklabels(labels,rotation=90)
    plt.tight_layout()
    ax.axhline(0,color='k',alpha=0.2)
    plt.ylim(ymax=5,ymin=-20)
    if not (type(filename) == type(None)):
        plt.savefig(filename+"_dropout.png")

def plot_summaries(psydata):
    '''
    Debugging function that plots the moving average of many behavior variables 
    '''
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




def process_session(experiment_id,complete=True,format_options={}):
    '''
        Fits the model, does bootstrapping for parameter recovery, and dropout analysis and cross validation
    
    '''
    print("Pulling Data")
    session = get_data(experiment_id)
    print("Annotating lick bouts")
    pt.annotate_licks(session) 
    pm.annotate_bouts(session)
    print("Formating Data")
    psydata = format_session(session,format_options)
    filename = global_directory + str(experiment_id)
    if not complete:
        if format_options['timing0/1']:
            directory="/home/alex.piet/codebase/behavior/psy_fits_v3_01/"
        else:
            directory="/home/alex.piet/codebase/behavior/psy_fits_v3_11/"
        filename = directory + str(experiment_id)

    print("Initial Fit")
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata)
    ypred,ypred_each = compute_ypred(psydata, wMode,weights)
    plot_weights(wMode, weights,psydata,errorbar=credibleInt, ypred = ypred,filename=filename)
    if complete:
        print("Bootstrapping")
        boots = bootstrap(10, psydata, ypred, weights, wMode)
        plot_bootstrap(boots, hyp, weights, wMode, credibleInt,filename=filename)
        print("Dropout Analysis")
        models, labels = dropout_analysis(psydata)
        plot_dropout(models,labels,filename=filename)
    print("Cross Validation Analysis")
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    cv_pred = compute_cross_validation_ypred(psydata, cross_results,ypred)
    try:
        metadata = session.metadata
    except:
        metadata = []
    if complete:
        output = [models,    labels,   boots,   hyp,   evd,   wMode,   hess,   credibleInt,   weights,   ypred,  psydata,  cross_results,  cv_pred,  metadata]
        labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
    else:
        output = [ hyp,   evd,   wMode,   hess,   credibleInt,   weights,   ypred,  psydata,  cross_results,  cv_pred,  metadata]
        labels = ['hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']       
    fit = dict((x,y) for x,y in zip(labels, output))
    fit['ID'] = experiment_id
    if complete:
        fit = cluster_fit(fit) # gets saved separately
    save(filename+".pkl", fit) 
    plt.close('all')

def plot_session_summary_priors(IDS,directory=None,savefig=False,group_label=""):
    '''
        Make a summary plot of the priors on each feature
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    alld = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e :
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
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

    if counter == 0:
        print('NO DATA')
        return

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


def plot_session_summary_correlation(IDS,directory=None,savefig=False,group_label="",verbose=True):
    '''
        Make a summary plot of the priors on each feature
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    scores = []
    ids = []
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e :
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
        else:
            fit = session_summary[7]
            r2 = compute_model_prediction_correlation(fit,fit_mov=25,data_mov=25,plot_this=False,cross_validation=True)
            scores.append(r2)
            ids.append(id)
            counter +=1

    if counter == 0:
        print('NO DATA')
        return


    ax.hist(np.array(scores),bins=50)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('$R^2$', fontsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    meanscore = np.median(np.array(scores))
    ax.plot(meanscore, ax.get_ylim()[1],'rv')
    ax.axvline(meanscore,color='r', alpha=0.3)
    ax.set_xlim(0,1)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"correlation.png")
    if verbose:
        median = np.argsort(np.array(scores))[len(scores)//2]
        best = np.argmax(np.array(scores))
        worst = np.argmin(np.array(scores)) 
        print('Worst  Session: ' + str(ids[worst]) + " " + str(scores[worst]))
        print('Median Session: ' + str(ids[median]) + " " + str(scores[median]))
        print('Best   Session: ' + str(ids[best]) + " " + str(scores[best]))      
    return scores, ids 

def plot_session_summary_dropout(IDS,directory=None,cross_validation=True,savefig=False,group_label="",model_evidence=False):
    '''
        Make a summary plot showing the fractional change in either model evidence (not cross-validated), or log-likelihood (cross-validated)
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(figsize=(7.2,6))
    alld = None
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory, cross_validation_dropout=cross_validation,model_evidence=model_evidence)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
        else:
            dropout = session_summary[2]
            labels  = session_summary[3]
            ax.plot(np.arange(0,len(dropout)),dropout, 'o',alpha=0.5)
            ax.set_xticks(np.arange(0,len(dropout)))
            ax.set_xticklabels(labels,fontsize=12, rotation = 90)
            if model_evidence:
                plt.ylabel('% Change in Normalized Model Evidence \n More Negative = Worse Fit',fontsize=12)
            else:
                plt.ylabel('% Change in normalized cross-validated likelihood \n More Negative = Worse Fit',fontsize=12)

            if type(alld) == type(None):
                alld = dropout
            else:
                alld += dropout
            counter +=1
    if counter == 0:
        print('NO DATA')
        return

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
        if model_evidence:
            plt.savefig(directory+"summary_"+group_label+"dropout_model_evidence.png")
        elif cross_validation:
            plt.savefig(directory+"summary_"+group_label+"dropout_cv.png")
        else:
            plt.savefig(directory+"summary_"+group_label+"dropout.png")

def plot_session_summary_weights(IDS,directory=None, savefig=False,group_label=""):
    '''
        Makes a summary plot showing the average weight value for each session
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    allW = None
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
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
    if counter == 0:
        print('NO DATA')
        return

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

def plot_session_summary_weight_range(IDS,directory=None,savefig=False,group_label=""):
    '''
        Makes a summary plot showing the range of each weight across each session
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    allW = None
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
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

    if counter == 0:
        print('NO DATA')
        return
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

def plot_session_summary_weight_scatter(IDS,directory=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against each other weight, plotting the average weight for each session
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=nel,ncols=nel,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory= directory)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
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
            counter +=1
    plt.tight_layout()
    if counter == 0:
        print('NO DATA')
        return
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_scatter.png")

def plot_session_summary_dropout_scatter(IDS,directory=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of the dropout performance change for each feature against each other feature 
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=3,ncols=3,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory, cross_validation_dropout=True)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
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
            counter+=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"dropout_scatter.png")


def plot_session_summary_weight_avg_scatter(IDS,directory=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against each other weight, plotting the average weight for each session
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=nel,ncols=nel,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
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
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_avg_scatter.png")

def plot_session_summary_weight_avg_scatter_task0(IDS,directory=None,savefig=False,group_label="",nel=3):
    '''
        Makes a summary plot of the average weights of task0 against omission weights for each session
        Also computes a regression line, and returns the linear model
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,6))
    allx = []
    ally = []
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
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
            counter+=1
    if counter == 0:
        print('NO DATA')
        return
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


def plot_session_summary_weight_avg_scatter_hits(IDS,directory=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of each weight against the total number of hits
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=2,ncols=5,figsize=(14,6))
    allW = None
    counter = 0
    xmax = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
        else:
            W = session_summary[6]
            fit = session_summary[7]
            hits = np.sum(fit['psydata']['hits'])
            xmax = np.max([hits, xmax])
            weights  = session_summary[1]
            weights_list = []
            for i in sorted(weights.keys()):
                weights_list += [i]*weights[i]
            for i in np.arange(0,np.shape(W)[0]):
                ax[0,i].axhline(0,color='k',alpha=0.1)
                meanWi = np.mean(W[i,:])
                stdWi = np.std(W[i,:])
                ax[0,i].plot([hits, hits], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax[0,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[0,i].set_xlabel('hits',fontsize=12)
                ax[0,i].set_ylabel(weights_list[i],fontsize=12)
                ax[0,i].xaxis.set_tick_params(labelsize=12)
                ax[0,i].yaxis.set_tick_params(labelsize=12)
                ax[0,i].set_xlim(xmin=0,xmax=xmax)

                meanWi = transform(np.mean(W[i,:]))
                stdWiPlus = transform(np.mean(W[i,:])+np.std(W[i,:]))
                stdWiMinus =transform(np.mean(W[i,:])-np.std(W[i,:])) 
                ax[1,i].plot([hits, hits], [stdWiMinus, stdWiPlus],'k-',alpha=0.1)
                ax[1,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[1,i].set_xlabel('hits',fontsize=12)
                ax[1,i].set_ylabel(weights_list[i],fontsize=12)
                ax[1,i].xaxis.set_tick_params(labelsize=12)
                ax[1,i].yaxis.set_tick_params(labelsize=12)
                ax[1,i].set_xlim(xmin=0,xmax=xmax)
                ax[1,i].set_ylim(ymin=0,ymax=1)

            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_avg_scatter_hits.png")

def plot_session_summary_weight_avg_scatter_false_alarms(IDS,directory=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of each weight against the total number of false_alarms
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=2,ncols=5,figsize=(14,6))
    allW = None
    counter = 0
    xmax = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
        else:
            W = session_summary[6]
            fit = session_summary[7]
            hits = np.sum(fit['psydata']['false_alarms'])
            xmax = np.max([hits, xmax])
            weights  = session_summary[1]
            weights_list = []
            for i in sorted(weights.keys()):
                weights_list += [i]*weights[i]
            for i in np.arange(0,np.shape(W)[0]):
                ax[0,i].axhline(0,color='k',alpha=0.1)
                meanWi = np.mean(W[i,:])
                stdWi = np.std(W[i,:])
                ax[0,i].plot([hits, hits], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax[0,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[0,i].set_xlabel('false_alarms',fontsize=12)
                ax[0,i].set_ylabel(weights_list[i],fontsize=12)
                ax[0,i].xaxis.set_tick_params(labelsize=12)
                ax[0,i].yaxis.set_tick_params(labelsize=12)
                ax[0,i].set_xlim(xmin=0,xmax=xmax)

                meanWi = transform(np.mean(W[i,:]))
                stdWiPlus = transform(np.mean(W[i,:])+np.std(W[i,:]))
                stdWiMinus =transform(np.mean(W[i,:])-np.std(W[i,:])) 
                ax[1,i].plot([hits, hits], [stdWiMinus, stdWiPlus],'k-',alpha=0.1)
                ax[1,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[1,i].set_xlabel('false_alarms',fontsize=12)
                ax[1,i].set_ylabel(weights_list[i],fontsize=12)
                ax[1,i].xaxis.set_tick_params(labelsize=12)
                ax[1,i].yaxis.set_tick_params(labelsize=12)
                ax[1,i].set_xlim(xmin=0,xmax=xmax)
                ax[1,i].set_ylim(ymin=0,ymax=1)

            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_avg_scatter_false_alarms.png")

def plot_session_summary_weight_avg_scatter_miss(IDS,directory=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of each weight against the total number of miss
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=2,ncols=5,figsize=(14,6))
    allW = None
    counter = 0
    xmax = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
        else:
            W = session_summary[6]
            fit = session_summary[7]
            hits = np.sum(fit['psydata']['misses'])
            xmax = np.max([hits, xmax])
            weights  = session_summary[1]
            weights_list = []
            for i in sorted(weights.keys()):
                weights_list += [i]*weights[i]
            for i in np.arange(0,np.shape(W)[0]):
                ax[0,i].axhline(0,color='k',alpha=0.1)
                meanWi = np.mean(W[i,:])
                stdWi = np.std(W[i,:])
                ax[0,i].plot([hits, hits], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax[0,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[0,i].set_xlabel('misses',fontsize=12)
                ax[0,i].set_ylabel(weights_list[i],fontsize=12)
                ax[0,i].xaxis.set_tick_params(labelsize=12)
                ax[0,i].yaxis.set_tick_params(labelsize=12)
                ax[0,i].set_xlim(xmin=0,xmax=xmax)

                meanWi = transform(np.mean(W[i,:]))
                stdWiPlus = transform(np.mean(W[i,:])+np.std(W[i,:]))
                stdWiMinus =transform(np.mean(W[i,:])-np.std(W[i,:])) 
                ax[1,i].plot([hits, hits], [stdWiMinus, stdWiPlus],'k-',alpha=0.1)
                ax[1,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[1,i].set_xlabel('misses',fontsize=12)
                ax[1,i].set_ylabel(weights_list[i],fontsize=12)
                ax[1,i].xaxis.set_tick_params(labelsize=12)
                ax[1,i].yaxis.set_tick_params(labelsize=12)
                ax[1,i].set_xlim(xmin=0,xmax=xmax)
                ax[1,i].set_ylim(ymin=0,ymax=1)

            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_avg_scatter_misses.png")

def plot_session_summary_weight_trajectory(IDS,directory=None,savefig=False,group_label="",nel=3):
    '''
        Makes a summary plot by plotting each weights trajectory across each session. Plots the average trajectory in bold
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=nel+1,ncols=1,figsize=(6,10))
    allW = None
    counter = 0
    xmax  =  []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
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
                allW = W[:,0:3800]
            else:
                allW += W[:,0:3800]
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
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
    '''
        computes the full log likelihood by summing each cross validation fold
    '''
    return np.sum([i['logli'] for i in cv_results]) 

def get_Excit_IDS(all_metadata):
    '''
        Given a list of metadata (get_all_metadata), returns a list of IDS with excitatory CRE lines
    '''
    IDS =[]
    for m in all_metadata:
        if m['full_genotype'][0:5] == 'Slc17':
            IDS.append(m['ophys_experiment_id'])
    return IDS

def get_Inhib_IDS(all_metadata):
    '''
        Given a list of metadata (get_all_metadata), returns a list of IDS with inhibitory CRE lines
    '''
    IDS =[]
    for m in all_metadata:
        if not( m['full_genotype'][0:5] == 'Slc17'):
            IDS.append(m['ophys_experiment_id'])
    return IDS

def get_stage_names(IDS):
    '''
        Compiles a list of the stage number for each ophys session
    '''
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


def get_all_metadata(IDS,directory=None):
    '''
        Compiles a list of metadata for every session in IDS
    '''
    if type(directory) == type(None):
        directory = global_directory
    m = []
    for id in IDS:
        try:
            filename = directory + str(id) + ".pkl" 
            fit = load(filename)
            if not (type(fit) == type(dict())):
                labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
                fit = dict((x,y) for x,y in zip(labels, fit))
            metadata = fit['metadata']
            m.append(metadata)
        except:
            pass
    
    return m
           
def get_session_summary(experiment_id,cross_validation_dropout=True,model_evidence=False,directory=None):
    '''
        Extracts useful summary information about each fit
        if cross_validation_dropout, then uses the dropout analysis where each reduced model is cross-validated
    '''
    if type(directory) == type(None):
        directory = global_directory

    filename = directory + str(experiment_id) + ".pkl" 
    fit = load(filename)
    if not (type(fit) == type(dict())) :
        labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
        fit = dict((x,y) for x,y in zip(labels, fit))
    # compute statistics
    dropout = []
    if model_evidence:
        for i in np.arange(0, len(fit['models'])):
            dropout.append(fit['models'][i][1] )
        dropout = np.array(dropout)
        dropout = (1-dropout/dropout[0])*100
    elif cross_validation_dropout:
        for i in np.arange(0, len(fit['models'])):
            dropout.append(get_cross_validation_dropout(fit['models'][i][6]))
        dropout = np.array(dropout)
        dropout = (1-dropout/dropout[0])*100
    else:
        for i in np.arange(0, len(fit['models'])):
            dropout.append((1-fit['models'][i][1]/fit['models'][0][1])*100)
        dropout = np.array(dropout)
    avgW = np.mean(fit['wMode'],1)
    rangeW = np.ptp(fit['wMode'],1)
    return fit['hyp']['sigma'],fit['weights'],dropout,fit['labels'], avgW, rangeW,fit['wMode'],fit

def plot_session_summary(IDS,directory=None,savefig=False,group_label="",nel=3):
    '''
        Makes a series of summary plots for all the IDS
    '''
    if type(directory) == type(None):
        directory = global_directory
    plot_session_summary_priors(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_dropout(IDS,directory=directory,cross_validation=False,savefig=savefig,group_label=group_label)
    plot_session_summary_dropout(IDS,directory=directory,cross_validation=True,savefig=savefig,group_label=group_label)
    plot_session_summary_dropout(IDS,directory=directory,model_evidence=True,savefig=savefig,group_label=group_label)
    plot_session_summary_dropout_scatter(IDS, directory=directory, savefig=savefig, group_label=group_label) # hard coded which to scatter
    plot_session_summary_weights(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_range(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_scatter(IDS,directory=directory,savefig=savefig,group_label=group_label,nel=nel)
    plot_session_summary_weight_avg_scatter(IDS,directory=directory,savefig=savefig,group_label=group_label,nel=nel)
    plot_session_summary_weight_avg_scatter_task0(IDS,directory=directory,savefig=savefig,group_label=group_label,nel=nel)
    plot_session_summary_weight_avg_scatter_hits(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_avg_scatter_miss(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_avg_scatter_false_alarms(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_weight_trajectory(IDS,directory=directory,savefig=savefig,group_label=group_label,nel=nel)
    plot_session_summary_logodds(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_correlation(IDS,directory=directory,savefig=savefig,group_label=group_label)
    plot_session_summary_roc(IDS,directory=directory,savefig=savefig,group_label=group_label)


def compute_cross_validation(psydata, hyp, weights,folds=10):
    '''
        Computes Cross Validation for the data given the regressors as defined in hyp and weights
    '''
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
    '''
        Computes the predicted outputs from cross validation results by stitching together the predictions from each folds test set
        full_pred is a vector of probabilities (0,1) for each time bin in psydata
    '''
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


def plot_session_summary_logodds(IDS,directory=None,savefig=False,group_label="",cross_validation=True):
    '''
        Makes a summary plot of the log-odds of the model fits = log(prob(lick|lick happened)/prob(lick|no lick happened))
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4.5))
    logodds=[]
    counter =0
    for id in IDS:
        try:
            #session_summary = get_session_summary(id)
            filenamed = directory + str(id) + ".pkl" 
            output = load(filenamed)
            if not (type(output) == type(dict())):
                labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
                fit = dict((x,y) for x,y in zip(labels, output))
            else:
                fit = output
        except Exception as e:
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
        else:
            if cross_validation:
                lickedp = np.mean(fit['cv_pred'][fit['psydata']['y'] ==2])
                nolickp = np.mean(fit['cv_pred'][fit['psydata']['y'] ==1])
            else:
                lickedp = np.mean(fit['ypred'][fit['psydata']['y'] ==2])
                nolickp = np.mean(fit['ypred'][fit['psydata']['y'] ==1])
            ax[0].plot(nolickp,lickedp, 'o', alpha = 0.5)
            logodds.append(np.log(lickedp/nolickp))
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
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
    meanscore = np.median(np.array(logodds))
    ax[1].plot(meanscore, ax[1].get_ylim()[1],'rv')
    ax[1].axvline(meanscore,color='r', alpha=0.3)


    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"weight_logodds.png")


def get_all_weights(IDS,directory=None):
    '''
        Returns a concatenation of all weights for every session in IDS
    '''
    if type(directory) == type(None):
        directory = global_directory
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

def load_fit(ID, directory=None):
    '''
        Loads the fit for session ID, in directory
        Creates a dictionary for the session
        if the fit has cluster labels then it loads them and puts them into the dictionary
    '''
    if type(directory) == type(None):
        directory = global_directory
    filename = directory + str(ID) + ".pkl" 
    output = load(filename)
    if not (type(output) == type(dict())):
        labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
        fit = dict((x,y) for x,y in zip(labels, output))
    else:
        fit = output
    fit['ID'] = ID
    #if os.path.isfile(directory+str(ID) + "_clusters.pkl"):
    #    clusters = load(directory+str(ID) + "_clusters.pkl")
    #    fit['clusters'] = clusters
    #else:
    #    fit = cluster_fit(fit,directory=directory)
    if os.path.isfile(directory+str(ID) + "_all_clusters.pkl"):
        fit['all_clusters'] = load(directory+str(ID) + "_all_clusters.pkl")
    return fit

def plot_cluster(ID, cluster, fit=None, directory=None):
    if type(directory) == type(None):
        directory = global_directory
    if not (type(fit) == type(dict())):
        fit = load_fit(ID, directory=directory)
    plot_fit(ID,fit=fit, cluster_labels=fit['clusters'][str(cluster)][1])

def plot_fit(ID, cluster_labels=None,fit=None, directory=None,validation=True,savefig=False,num_clusters=None):
    '''
        Plots the fit associated with a session ID
        Needs the fit dictionary. If you pass these values into, the function is much faster 
    '''
    if type(directory) == type(None):
        directory = global_directory
    if not (type(fit) == type(dict())):
        fit = load_fit(ID, directory=directory)
    if savefig:
        filename = directory + str(ID)
    else:
        filename=None
    plot_weights(fit['wMode'], fit['weights'],fit['psydata'],errorbar=fit['credibleInt'], ypred = fit['ypred'],cluster_labels=cluster_labels,validation=validation,filename=filename,num_clusters=num_clusters)
    return fit
   
def cluster_fit(fit,directory=None,minC=2,maxC=4):
    '''
        Given a fit performs a series of clustering, adds the results to the fit dictionary, and saves the results to a pkl file
    '''
    if type(directory) == type(None):
        directory = global_directory
    numc= range(minC,maxC+1)
    cluster = dict()
    for i in numc:
        output = cluster_weights(fit['wMode'],i)
        cluster[str(i)] = output
    fit['cluster'] = cluster
    filename = directory + str(fit['ID']) + "_clusters.pkl" 
    save(filename, cluster) 
    return fit

def cluster_weights(wMode,num_clusters):
    '''
        Clusters the weights in wMode into num_clusters clusters
    '''
    output = k_means(transform(wMode.T),num_clusters)
    return output

def check_clustering(wMode,numC=5):
    '''
        For a set of weights (regressors x time points), computes a series of clusterings from 1 up to numC clusters
        Plots the weights and the cluster labelings
        
        Returns the scores for each clustering
    '''
    fig,ax = plt.subplots(nrows=numC,ncols=1)
    scores = []
    for j in range(0,numC):
        for i in range(0,4):
            ax[j].plot(transform(wMode[i,:]))
        output = cluster_weights(wMode,j+1)
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
    '''
        For each session in IDS, performs clustering from 1 cluster up to numC clusters
        Plots the normalized error (euclidean distance from each point to each cluster center) for each cluster-number
    '''
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
                    output = cluster_weights(wMode,j+1)
                    scores.append(output[2])
                all_scores.append(scores)
    
    plt.figure()
    for i in np.arange(0,len(all_scores)):
        plt.plot(np.arange(1,j+2), all_scores[i]/all_scores[i][0],'k-',alpha=0.3)    
    plt.ylabel('Normalized error')
    plt.xlabel('number of clusters')
    

def load_mouse(mouse,get_ophys=True, get_behavior=False):
    '''
        Takes a mouse donor_id, and filters the sessions in Nick's database, and returns a list of session objects. Optional arguments filter what types of sessions are returned    
    '''
    manifest = get_manifest()
    mouse_manifest = manifest[manifest['animal_name'] == int(mouse)]
    mouse_manifest = mouse_manifest.sort_values(by='date_of_acquisition')
 
    #vb_sessions = pd.read_hdf('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/vb_sessions.h5', key='df')
    #vb_sessions_good = vb_sessions[vb_sessions['stage_name'] != 'Load error']
    #mouse_session =  vb_sessions_good[vb_sessions_good['donor_id'] == mouse]  
 
    sessions = []
    IDS = []
    active =[]
    for index, row in mouse_manifest.iterrows():
        session, session_id = load_session(row,get_ophys=get_ophys, get_behavior=get_behavior)
        if not (type(session) == type(None)): 
            sessions.append(session)
            IDS.append(session_id)
            active.append(not parse_stage_name_for_passive(row.stage_name))
    return sessions,IDS,active

def load_session(row,get_ophys=True, get_behavior=False):
    '''
        Takes in a row of Nick's database of sessions and loads a session either via the ophys interface or behavior interface. Two optional arguments toggle what types of data are returned 
    '''
    if pd.isnull(row['ophys_experiment_id']):
        session_id = 'behavior_{}'.format(int(row['behavior_session_id']))
        if get_behavior:
            # this will crash because we haven't supported bs yet
            api = bla.BehaviorLimsApi(int(row['behavior_session_id']))
            session = bs.BehaviorSession(api)
        else:
            session = None
    else:
        session_id = 'ophys_{}'.format(int(row['ophys_experiment_id']))
        if get_ophys:
            session = get_data(int(row['ophys_experiment_id']))
            session.metadata['stage'] = row.stage_name
        else:
            session = None
    return session, session_id

def format_mouse(sessions,IDS):
    '''
        Takes a list of sessions and returns a list of psydata formatted dictionaries for each session, and IDS a list of the IDS that go into each session
    '''
    d =[]
    good_ids =[]
    for session, id in zip(sessions,IDS):
        try:
            pt.annotate_licks(session) 
            pm.annotate_bouts(session)
            psydata = format_session(session)
        except Exception as e:
            print(str(id) +" "+ str(e))
        else:
            print(str(id))
            d.append(psydata)
            good_ids.append(id)
    return d, good_ids

def merge_datas(psydatas):
    ''' 
        Takes a list of psydata dictionaries and concatenates them into one master dictionary. Computes the dayLength field to keep track of where day-breaks are
        Also records the session_label for each dictionary
    '''
    if len(psydatas) == 0:
        raise Exception('No data to merge')
    if len(psydatas) == 1:
        print('Only one session, no need to merge')
        psydata = psydatas[0]
        return psydata
    else:
        print('Merging ' + str(len(psydatas)) + ' sessions')
    psydata = copy.deepcopy(psydatas[0])
    psydata['dayLength'] = [len(psydatas[0]['y'])]
    for d in psydatas[1:]:    
        psydata['y'] = np.concatenate([psydata['y'], d['y']])
        for key in psydata['inputs'].keys():
            psydata['inputs'][key] = np.concatenate([psydata['inputs'][key], d['inputs'][key]])

        psydata['false_alarms'] = np.concatenate([psydata['false_alarms'], d['false_alarms']])
        psydata['correct_reject'] = np.concatenate([psydata['correct_reject'], d['correct_reject']])
        psydata['hits'] = np.concatenate([psydata['hits'], d['hits']])
        psydata['misses'] = np.concatenate([psydata['misses'], d['misses']])
        psydata['aborts'] = np.concatenate([psydata['aborts'], d['aborts']])
        psydata['auto_rewards'] = np.concatenate([psydata['auto_rewards'], d['auto_rewards']])
        psydata['start_times'] = np.concatenate([psydata['start_times'], d['start_times']])
        psydata['session_label']= np.concatenate([psydata['session_label'], d['session_label']])
        psydata['dayLength'] = np.concatenate([psydata['dayLength'], [len(d['y'])]])
        psydata['flash_ids'] = np.concatenate([psydata['flash_ids'],d['flash_ids']])
        psydata['df'] = pd.concat([psydata['df'], d['df']])

    psydata['dayLength'] = np.array(psydata['dayLength'])
    return psydata


def process_mouse(donor_id):
    '''
        Takes a mouse donor_id, loads all ophys_sessions, and fits the model in the temporal order in which the data was created. Does not do cross validation 
    '''
    print('Building List of Sessions and pulling')
    sessions, all_IDS,active = load_mouse(donor_id) # sorts the sessions by time
    print('Got  ' + str(len(all_IDS)) + ' sessions')
    print("Formating Data")
    psydatas, good_IDS = format_mouse(np.array(sessions)[active],np.array(all_IDS)[active])
    print('Got  ' + str(len(good_IDS)) + ' good sessions')
    print("Merging Formatted Sessions")
    psydata = merge_datas(psydatas)
    filename = global_directory + 'mouse_' + str(donor_id) 

    print("Initial Fit")    
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,OMISSIONS=True)
    ypred,ypred_each = compute_ypred(psydata, wMode,weights)
    plot_weights(wMode, weights,psydata,errorbar=credibleInt, ypred = ypred,filename=filename, session_labels = psydata['session_label'])

    print("Cross Validation Analysis")
    cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    cv_pred = compute_cross_validation_ypred(psydata, cross_results,ypred)

    metadata =[]
    for s in sessions:
        try:
            m = s.metadata
        except:
            m = []
        metadata.append(m)

    labels = ['hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','good_IDS','metadata','all_IDS','active','cross_results','cv_pred','mouse_ID']
    output = [hyp, evd, wMode, hess, credibleInt, weights, ypred,psydata,good_IDS,metadata,all_IDS,active,cross_results,cv_pred,donor_id]
    fit = dict((x,y) for x,y in zip(labels, output))
   
    print("Clustering Behavioral Epochs")
    fit = cluster_mouse_fit(fit)

    save(filename+".pkl", fit)
    plt.close('all')

def get_all_ophys_IDS():
    '''
        Returns a list of all unique ophys_sessions in Nick's database of sessions
    '''
    vb_sessions = pd.read_hdf('/home/nick.ponvert/data/vb_sessions.h5', key='df')
    vb_sessions_good = vb_sessions[vb_sessions['stage_name'] != 'Load error']
    all_ids = vb_sessions_good[~vb_sessions_good['ophys_experiment_id'].isnull()]['ophys_experiment_id'].values
    IDS=[]
    for id in all_ids:
        IDS.append(int(id))
    return IDS

def get_all_mice():
    '''
        Returns a list of all unique mice donor_ids in Nick's database of sessions
    '''
    vb_sessions = pd.read_hdf('/home/nick.ponvert/data/vb_sessions.h5', key='df')
    vb_sessions_good = vb_sessions[vb_sessions['stage_name'] != 'Load error']
    mice = np.unique(vb_sessions_good['donor_id'])
    return mice

def get_good_behavior_IDS(IDS,hit_threshold=100):
    '''
        Filters all the ids in IDS for sessions with greather than hit_threshold hits
        Returns a list of session ids
    '''
    good_ids = []
    for id in IDS:
        try:
            summary = get_session_summary(id)
        except:
            pass
        else:
            if np.sum(summary[7]['psydata']['hits']) > hit_threshold:
                good_ids.append(id)
    return good_ids

def compute_model_prediction_correlation(fit,fit_mov=50,data_mov=50,plot_this=False,cross_validation=True):
    '''
        Computes the R^2 value between the model predicted licking probability, and the smoothed data lick rate.
        The data is smoothed over data_mov flashes. The model is smoothed over fit_mov flashes. Both smoothings uses a moving _mean within that range. 
        if plot_this, then the two smoothed traces are plotted
        if cross_validation, then uses the cross validated model prediction, and not the training set predictions
        Returns, the r^2 value.
    '''
    if cross_validation:
        data = copy.copy(fit['psydata']['y']-1)
        model = copy.copy(fit['cv_pred'])
    else:
        data = copy.copy(fit['psydata']['y']-1)
        model = copy.copy(fit['ypred'])
    data_smooth = moving_mean(data,data_mov)
    ypred_smooth = moving_mean(model,fit_mov)

    minlen = np.min([len(data_smooth), len(ypred_smooth)])
    if plot_this:
        plt.figure()
        plt.plot(ypred_smooth, 'k')
        plt.plot(data_smooth,'b')
    return round(np.corrcoef(ypred_smooth[0:minlen], data_smooth[0:minlen])[0,1]**2,2)

def compute_model_roc(fit,plot_this=False,cross_validation=True):
    '''
        Computes area under the ROC curve for the model in fit. If plot_this, then plots the ROC curve. 
        If cross_validation, then uses the cross validated prediction in fit, not he training fit.
        Returns the AU. ROC single float
    '''
    if cross_validation:
        data = copy.copy(fit['psydata']['y']-1)
        model = copy.copy(fit['cv_pred'])
    else:
        data = copy.copy(fit['psydata']['y']-1)
        model = copy.copy(fit['ypred'])

    if plot_this:
        plt.figure()
        alarms,hits,thresholds = roc_curve(data,model)
        plt.plot(alarms,hits,'ko-')
        plt.plot([0,1],[0,1],'k--')
        plt.ylabel('Hits')
        plt.xlabel('False Alarms')
    return roc_auc_score(data,model)

def plot_session_summary_roc(IDS,directory=None,savefig=False,group_label="",verbose=True,cross_validation=True):
    '''
        Make a summary plot of the histogram of AU.ROC values for all sessions in IDS.
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    scores = []
    ids = []
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except Exception as e :
            if not (str(e)[0:35] == '[Errno 2] No such file or directory'):
                print(str(e))
        else:
            fit = session_summary[7]
            roc = compute_model_roc(fit,plot_this=False,cross_validation=cross_validation)
            scores.append(roc)
            ids.append(id)
            counter +=1

    if counter == 0:
        print('NO DATA')
        return
    ax.set_xlim(0.5,1)
    ax.hist(np.array(scores),bins=25)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('ROC-AUC', fontsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    meanscore = np.median(np.array(scores))
    ax.plot(meanscore, ax.get_ylim()[1],'rv')
    ax.axvline(meanscore,color='r', alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"summary_"+group_label+"roc.png")
    if verbose:
        median = np.argsort(np.array(scores))[len(scores)//2]
        best = np.argmax(np.array(scores))
        worst = np.argmin(np.array(scores)) 
        print('Worst  Session: ' + str(ids[worst]) + " " + str(scores[worst]))
        print('Median Session: ' + str(ids[median]) + " " + str(scores[median]))
        print('Best   Session: ' + str(ids[best]) + " " + str(scores[best]))      
    return scores, ids 

def load_mouse_fit(ID, directory=None):
    '''
        Loads the fit for session ID, in directory
        Creates a dictionary for the session
        if the fit has cluster labels then it loads them and puts them into the dictionary
    '''
    if type(directory) == type(None):
        directory = global_directory

    filename = directory + "mouse_"+ str(ID) + ".pkl" 
    fit = load(filename)
    fit['mouse_ID'] = ID
    #if os.path.isfile(directory+"mouse_"+str(ID) + "_clusters.pkl"):
    #    clusters = load(directory+"mouse_"+str(ID) + "_clusters.pkl")
    #    fit['clusters'] = clusters
    #else:
    #    fit = cluster_mouse_fit(fit,directory=directory)
    return fit


def cluster_mouse_fit(fit,directory=None,minC=2,maxC=4):
    '''
        Given a fit performs a series of clustering, adds the results to the fit dictionary, and saves the results to a pkl file
    '''
    if type(directory) == type(None):
        directory = global_directory

    numc= range(minC,maxC+1)
    cluster = dict()
    for i in numc:
        output = cluster_weights(fit['wMode'],i)
        cluster[str(i)] = output
    fit['cluster'] = cluster
    filename = directory + "mouse_" + str(fit['mouse_ID']) + "_clusters.pkl" 
    save(filename, cluster) 
    return fit

def plot_mouse_fit(ID, cluster_labels=None, fit=None, directory=None,validation=True,savefig=False):
    '''
        Plots the fit associated with a session ID
        Needs the fit dictionary. If you pass these values into, the function is much faster 
    '''
    if type(directory) == type(None):
        directory = global_directory

    if not (type(fit) == type(dict())):
        fit = load_mouse_fit(ID, directory=directory)
    if savefig:
        filename = directory + 'mouse_' + str(ID) 
    else:
        filename=None
    plot_weights(fit['wMode'], fit['weights'],fit['psydata'],errorbar=fit['credibleInt'], ypred = fit['ypred'],cluster_labels=cluster_labels,validation=validation,filename=filename,session_labels=fit['psydata']['session_label'])
    return fit

def get_all_fit_weights(ids,directory=None):
    '''
        Returns a list of all the regression weights for the sessions in IDS
        
        INPUTS:
        ids, a list of sessions
        
        OUTPUTS:
        w, a list of the weights in each session
        w_ids, the ids that loaded and have weights in w
    '''
    w = []
    w_ids = []
    for id in ids:
        print(id)
        try:
            fit = load_fit(id,directory)
            w.append(fit['wMode'])
            w_ids.append(id)
            print(" good")
        except:
            print(" crash")
            pass
    return w, w_ids

def merge_weights(w): 
    '''
        Merges a list of weights into one long array of weights
    '''
    return np.concatenate(w,axis=1)           

def cluster_all(w,minC=2, maxC=4,directory=None,save_results=False):
    '''
        Clusters the weights in array w. Uses the cluster_weights function
        
        INPUTS:
        w, an array of weights
        minC, the smallest number of clusters to try
        maxC, the largest number of clusters to try
        directory, where to save the results
    
        OUTPUTS:
        cluster, the output from cluster_weights
        
        SAVES:
        the cluster results in 'all_clusters.pkl'
    '''
    if type(directory) == type(None):
        directory = global_directory

    numc= range(minC,maxC+1)
    cluster = dict()
    for i in numc:
        output = cluster_weights(w,i)
        cluster[str(i)] = output
    if save_results:
        filename = directory + "all_clusters.pkl" 
        save(filename, cluster) 
    return cluster

def unmerge_cluster(cluster,w,w_ids,directory=None,save_results=False):
    '''
        Unmerges an array of weights and clustering results into a list for each session
        
        INPUTS:
        cluster, the clustering results from cluster_all
        w, an array of weights
        w_ids, the list of ids which went into w
    
        outputs,
        session_clusters, a list of cluster results on a session by session basis
    '''
    session_clusters = dict()
    counter = 0
    for weights, id in zip(w,w_ids):
        session_clusters[id] = dict()
        start = counter
        end = start + np.shape(weights)[1]
        for key in cluster.keys():
            session_clusters[id][key] =(cluster[key][0],cluster[key][1][start:end],cluster[key][2]) 
        counter = end
    if save_results:
        save_session_clusters(session_clusters,directory=directory)
        save_all_clusters(w_ids,session_clusters,directory=directory)
    return session_clusters

def save_session_clusters(session_clusters, directory=None):
    '''
        Saves the session_clusters in 'session_clusters,pkl'

    '''
    if type(directory) == type(None):
        directory = global_directory

    filename = directory + "session_clusters.pkl"
    save(filename,session_clusters)

def save_all_clusters(w_ids,session_clusters, directory=None):
    '''
        Saves each sessions all_clusters
    '''
    if type(directory) == type(None):
        directory = global_directory

    for key in session_clusters.keys():
        filename = directory + str(key) + "_all_clusters.pkl" 
        save(filename, session_clusters[key]) 

def build_all_clusters(ids,directory=None,save_results=False):
    '''
        Clusters all the sessions in IDS jointly
    '''
    if type(directory) == type(None):
        directory = global_directory
    w,w_ids = get_all_fit_weights(ids,directory=directory)
    w_all = merge_weights(w)
    cluster = cluster_all(w_all,directory=directory,save_results=save_results)
    session_clusters= unmerge_cluster(cluster,w,w_ids,directory=directory,save_results=save_results)

def check_session(ID, directory=None):
    '''
        Checks if the ID has a model fit computed
    '''
    if type(directory) == type(None):
        directory = global_directory

    filename = directory + str(ID) + ".pkl" 
    has_fit =  os.path.isfile(filename)

    if has_fit:
        print("Session has a fit, load the results with load_fit(ID)")
    else:
        print("Session does not have a fit, fit the session with process_session(ID)")
    return has_fit

def get_session_id(experiment_id):
    if INTERNAL:
        return experiment_id
    else:
        manifest = get_manifest()
        return manifest[manifest['ophys_experiment_id'] == experiment_id]['ophys_session_id'].values 

def get_experiment_id(session_id):
    manifest = get_manifest()
    return manifest[manifest['ophys_session_id'] == session_id]['ophys_experiment_id'].values 

def get_cache():
    if INTERNAL:
        cache_json = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/cache_20190813'
        cache = bpc.BehaviorProjectCache(cache_json)
        return cache
    else:
        return bpc.InternalCacheFromLims()

def get_manifest():
    cache = get_cache()
    if INTERNAL:
        manifest = cache.experiment_table
    else:
        manifest = cache.get_sessions()
    return manifest

def parse_stage_name_for_passive(stage_name):
    return stage_name[-1] == "e"

def get_intersection(list_of_ids):
    return reduce(np.intersect1d,tuple(list_of_ids))

def get_active_A_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[(~manifest['passive_session']) &(manifest['image_set']=='A')].ophys_experiment_id.values)
    return session_ids

def get_active_B_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[(~manifest['passive_session']) &(manifest['image_set']=='B')].ophys_experiment_id.values)
    return session_ids

def get_stage_ids(stage):
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['stage_name'].str[6] == str(stage)].ophys_experiment_id.values)
    return session_ids

def get_active_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[~manifest['passive_session']].ophys_experiment_id.values)
    return session_ids

def get_passive_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['passive_session']].ophys_experiment_id.values)
    return session_ids

def get_A_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['image_set'] == 'A'].ophys_experiment_id.values)
    return session_ids

def get_B_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['image_set'] == 'B'].ophys_experiment_id.values)
    return session_ids

def get_slc_session_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['cre_line'] == 'Slc17a7-IRES2-Cre'].ophys_experiment_id.values)
    return session_ids

def get_vip_session_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['cre_line'] == 'Vip-IRES-Cre'].ophys_experiment_id.values)
    return session_ids

def get_session_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest.ophys_experiment_id.values)
    return session_ids

def get_mice_ids():
    manifest = get_manifest()
    mice_ids = np.unique(manifest.animal_name.values)
    return mice_ids
    
def get_mice_sessions(mouse_id):
    manifest = get_manifest()
    mouse_manifest = manifest[manifest['animal_name'] == int(mouse_id)]
    mouse_manifest = mouse_manifest.sort_values(by='date_of_acquisition')
    return mouse_manifest.ophys_experiment_id.values

def get_all_dropout(IDS,directory=None): 
    '''
        For each session in IDS, returns the vector of dropout scores for each model
    '''
    # Add to big matr
    if type(directory) == type(None):
        directory = global_directory
    all_dropouts = []
    hits = []
    false_alarms = []
    misses = []
    # Loop through IDS
    for id in IDS:
        print(id)
        try:
            fit = load_fit(id,directory=directory)
            # from fit extract dropout scores
            dropout = np.empty((len(fit['models']),))
            for i in range(0,len(fit['models'])):
                dropout[i] = (1-fit['models'][i][1]/fit['models'][0][1])*100
            all_dropouts.append(dropout)
            hits.append(np.sum(fit['psydata']['hits']))
            false_alarms.append(np.sum(fit['psydata']['false_alarms']))
            misses.append(np.sum(fit['psydata']['misses']))
        except:
            print(" crash")
    dropouts = np.stack(all_dropouts,axis=1)
    filepath = directory + "all_dropouts.pkl"
    save(filepath, dropouts)
    return dropouts,hits, false_alarms, misses

def load_all_dropout(directory=None):
    dropout = load(directory+"all_dropouts.pkl")
    return dropout

def get_mice_dropout(mice_ids,directory=None):
    if type(directory) == type(None):
        directory = global_directory
    mice_dropouts = []
    mice_good_ids = []
    # Loop through IDS
    for id in mice_ids:
        print(id)
        this_mouse = []
        for sess in get_mice_sessions(id):
            print(" "+str(sess))
            try:
                fit = load_fit(sess,directory=directory)
                dropout = np.empty((len(fit['models']),))
                for i in range(0,len(fit['models'])):
                    dropout[i] = (1-fit['models'][i][1]/fit['models'][0][1])*100
                this_mouse.append(dropout)
            except:
                print(" crash")
        if len(this_mouse) > 0:
            this_mouse = np.stack(this_mouse,axis=1)
            mice_dropouts.append(this_mouse)
            mice_good_ids.append(id)
    return mice_dropouts,mice_good_ids


def PCA_on_dropout(dropouts,labels=None,mice_dropouts=None, mice_ids = None,hits=None,false_alarms=None, misses=None,directory=None):
    # get labels from fit['labels'] for random session
    # mice_dropouts, mice_good_ids = ps.get_mice_dropout(ps.get_mice_ids())
    # dropouts = ps.load_all_dropout()
    if type(directory) == type(None):
        directory = global_directory   
    dex = -(dropouts[2,:] - dropouts[16,:])
    pca = PCA()
    pca.fit(dropouts.T)
    X = pca.transform(dropouts.T)
    fig, ax = plt.subplots(2,1,figsize=(8,6))
    ax[0].axhline(0,color='k',alpha=0.2)
    ax[0].axvline(0,color='k',alpha=0.2)
    scat = ax[0].scatter(X[:,0], X[:,1],c=-dex,cmap='plasma')
    cbar = fig.colorbar(scat, ax = ax[0])
    cbar.ax.set_ylabel('Dropout % \n (Task - Timing)',fontsize=12)
    ax[0].set_xlabel('<-- more timing  -   PC 1  -   more task -->',fontsize=12)
    ax[0].set_ylabel('PC 2',fontsize=12)

    ax[1].axhline(0,color='k',alpha=0.2)
    for i in np.arange(0,len(dropouts)):
        if np.mod(i,2) == 0:
            ax[1].axvspan(i-.5,i+.5,color='k', alpha=0.1)
    pca1varexp = str(100*round(pca.explained_variance_ratio_[0],2))
    pca2varexp = str(100*round(pca.explained_variance_ratio_[1],2))
    ax[1].plot(pca.components_[0,:],'ko-',label='PC1 '+pca1varexp+"%")
    #ax[1].plot(pca.components_[1,:],'bo-',label='PC2 '+pca2varexp+"%")
    ax[1].set_xlabel('Model Component',fontsize=12)
    ax[1].set_ylabel('% change in \n evidence',fontsize=12)
    ax[1].tick_params(axis='both',labelsize=10)
    ax[1].set_xticks(np.arange(0,len(dropouts)))
    if type(labels) is not type(None):    
        ax[1].set_xticklabels(labels,rotation=90)
    ax[1].legend()
    plt.tight_layout()
    ax[0].set_ylim([-10,10])
    ax[0].set_xlim([-20,20])
    plt.savefig(directory+"dropout_pca.png")

    fig, ax = plt.subplots(2,3,figsize=(10,6))
    ax[0,0].axhline(0,color='k',alpha=0.2)
    ax[0,0].axvline(0,color='k',alpha=0.2)
    ax[0,0].scatter(X[:,0], dex,c=-dex,cmap='plasma')
    ax[0,0].set_xlabel('<-- more timing  -   PC 1  -   more task -->',fontsize=12)
    ax[0,0].set_ylabel('Task - Timing',fontsize=12)
    ax[0,0].set_ylim([-20,20])
    ax[0,0].set_xlim([-20,20])
    ax[0,1].plot(pca.explained_variance_ratio_*100,'ko-')
    ax[0,1].set_xlabel('PC Dimension',fontsize=12)
    ax[0,1].set_ylabel('Explained Variance %',fontsize=12)

    if type(mice_dropouts) is not type(None):
        ax[1,0].axhline(0,color='k',alpha=0.2)
        ax[1,0].set_ylabel('Task - Timing', fontsize=12)
        ax[1,0].set_xticks(range(0,len(mice_dropouts)))

        mean_drop = []
        for i in range(0, len(mice_dropouts)):
            mean_drop.append(-1*np.nanmean(mice_dropouts[i][2,:]-mice_dropouts[i][16,:]))
        sortdex = np.argsort(np.array(mean_drop))
        #mice_dropouts = np.array(mice_dropouts)[sortdex]
        mice_dropouts = [mice_dropouts[i] for i in sortdex]
        mean_drop = np.array(mean_drop)[sortdex]
        for i in range(0,len(mice_dropouts)):
            if np.mod(i,2) == 0:
                ax[1,0].axvspan(i-.5,i+.5,color='k', alpha=0.1)
            mouse_dex = -(mice_dropouts[i][2,:]-mice_dropouts[i][16,:])
            ax[1,0].scatter(i*np.ones(np.shape(mouse_dex)), mouse_dex,c=-mouse_dex,cmap='plasma',vmin=(-dex).min(),vmax=(-dex).max(),alpha=1)
            #ax[1,0].plot([i-0.5, i+0.5], [mean_drop[i],mean_drop[i]], 'k-',alpha=0.2)
        sorted_mice_ids = [mice_ids[i] for i in sortdex]
        ax[1,0].set_xticklabels(sorted_mice_ids,{'fontsize':10},rotation=90)
    if type(hits) is not type(None):
        ax[1,1].scatter(dex, hits,c=-dex,cmap='plasma')
        ax[1,1].set_ylabel('Hits/session',fontsize=12)
        ax[1,1].set_xlabel('Task-Timing',fontsize=12)
        ax[1,1].axvline(0,color='k',alpha=0.2)
        ax[1,1].set_xlim(-20,20)
        ax[1,1].set_ylim(bottom=0)

        ax[0,2].scatter(dex, false_alarms,c=-dex,cmap='plasma')
        ax[0,2].set_ylabel('FA/session',fontsize=12)
        ax[0,2].set_xlabel('Task-Timing',fontsize=12)
        ax[0,2].axvline(0,color='k',alpha=0.2)
        ax[0,2].set_xlim(-20,20)
        ax[0,2].set_ylim(bottom=0)


        ax[1,2].scatter(dex, misses,c=-dex,cmap='plasma')
        ax[1,2].set_ylabel('Miss/session',fontsize=12)
        ax[1,2].set_xlabel('Task-Timing',fontsize=12)
        ax[1,2].axvline(0,color='k',alpha=0.2)
        ax[1,2].set_xlim(-20,20)
        ax[1,2].set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(directory+"dropout_pca_2.png")
    return pca

def compare_fits(ID, directories):
    fits = []
    roc = []
    for d in directories:
        print(d)
        fits.append(load_fit(ID,directory=d))
        roc.append(compute_model_roc(fits[-1]))
    return fits,roc
    
def compare_all_fits(IDS, directories):
    all_fits = []
    all_roc = []
    all_ids = []
    for id in IDS:
        print(id)
        try:
            fits, roc = compare_fits(id,directories)
            all_fits.append(fits)
            all_roc.append(roc)
            all_ids.append(id)
        except:
            print(" crash")
    filename = directories[1] + "all_roc.pkl"
    save(filename,[all_ids,all_roc])
    return all_roc

def segment_mouse_fit(fit):
    # Takes a fit over many sessions
    # Returns a list of fit dictionaries for each session
    lengths = fit['psydata']['dayLength']
    indexes = np.cumsum(np.concatenate([[0],lengths]))
    fit['wMode_session'] = []
    fit['credibleInt_session'] = []
    fit['ypred_session'] = []
    fit['cv_pred_session'] = []
    fit['psydata_session'] = []
    for i in range(0, len(fit['psydata']['dayLength'])):
        w = fit['wMode'][:,indexes[i]:indexes[i+1]]
        fit['wMode_session'].append(w)
        w = fit['credibleInt'][:,indexes[i]:indexes[i+1]]
        fit['credibleInt_session'].append(w)
        w = fit['ypred'][indexes[i]:indexes[i+1]]
        fit['ypred_session'].append(w)
        w = fit['cv_pred'][indexes[i]:indexes[i+1]]
        fit['cv_pred_session'].append(w)
        w = fit['psydata']['y'][indexes[i]:indexes[i+1]]
        fit['psydata_session'].append(w)

def compare_roc_session_mouse(fit,directory):
    # Asking how different the ROC fits are with mouse fits
    fit['roc_session_individual'] = []
    for id in fit['good_IDS']:
        print(id)
        try:
            sfit = load_fit(id[6:],directory=directory)
            data = copy.copy(sfit['psydata']['y']-1)
            model =copy.copy(sfit['cv_pred'])
            fit['roc_session_individual'].append(roc_auc_score(data,model))
        except:
            fit['roc_session_individual'].append(np.nan)
        
def mouse_roc(fit):
    fit['roc_session'] = []
    for i in range(0,len(fit['psydata']['dayLength'])):
        data = copy.copy(fit['psydata_session'][i]-1)
        model = copy.copy(fit['cv_pred_session'][i])
        fit['roc_session'].append(roc_auc_score(data,model))

def get_all_mouse_roc(IDS,directory=None):
    labels = []
    rocs=[]
    for id in IDS:
        print(id)
        try:
            fit = load_mouse_fit(id,directory=directory)
            segment_mouse_fit(fit)
            mouse_roc(fit)
            rocs.append(fit['roc_session'])
            labels.append(fit['psydata']['session_label'])
        except:
            pass
    return labels, rocs

def compare_all_mouse_session_roc(IDS,directory=None):
    mouse_rocs = []
    session_rocs=[]
    for id in IDS:
        print(id)
        try:
            fit = load_mouse_fit(id,directory=directory)
            segment_mouse_fit(fit)
            mouse_roc(fit)
            compare_roc_session_mouse(fit,directory=directory) 
        except:
            print(" crash")
        else:
            mouse_rocs += fit['roc_session']
            session_rocs += fit['roc_session_individual']
    save(directory+"all_roc_session_mouse.pkl",[mouse_rocs,session_rocs])
    return mouse_rocs, session_rocs

def plot_all_mouse_session_roc(directory):
    rocs = load(directory+"all_roc_session_mouse.pkl")
    plt.figure()
    plt.plot(np.array(rocs[1])*100, np.array(rocs[0])*100,'ko')
    plt.plot([60,100],[60,100],'k--')
    plt.xlabel('Session ROC (%)')
    plt.ylabel('Mouse ROC (%)')
    plt.savefig(directory+"all_roc_session_mouse.png") 

def compare_mouse_roc(IDS, dir1, dir2):
    mouse_rocs1 = []
    mouse_rocs2 = []
    for id in IDS:
        print(id)
        try:
            fit1 = load_mouse_fit(id, directory=dir1)
            fit2 = load_mouse_fit(id, directory=dir2)
            segment_mouse_fit(fit1)
            segment_mouse_fit(fit2)
            mouse_roc(fit1)
            mouse_roc(fit2)
            mouse_rocs1+=fit1['roc_session']
            mouse_rocs2+=fit2['roc_session']        
        except:
            print(" crash")
    save(dir1+"all_roc_mouse_comparison.pkl",[mouse_rocs1,mouse_rocs2])
    return mouse_rocs1,mouse_rocs2

def plot_mouse_roc_comparisons(directory,label1="", label2=""):
    rocs = load(directory + "all_roc_mouse_comparison.pkl")
    plt.figure(figsize=(5.75,5))
    plt.plot(np.array(rocs[1])*100, np.array(rocs[0])*100,'ko')
    plt.plot([50,100],[50,100],'k--')
    plt.xlabel(label2+' ROC (%)')
    plt.ylabel(label1+' ROC (%)')
    plt.ylim([50,100])
    plt.xlim([50,100])
    plt.savefig(directory+"all_roc_mouse_comparison.png")






 
