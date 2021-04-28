import os
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import psytrack as psy
from psytrack.helper.crossValidation import split_data
from psytrack.helper.crossValidation import xval_loglike
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV as logregcv
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
import psy_timing_tools as pt
import psy_metrics_tools as pm
import psy_general_tools as pgt
from scipy.stats import norm
from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import psy_style as pstyle

global_directory= '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_dev/' 
root_directory  = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/'

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
  
def get_directory(version,verbose=False):
    if version is None:
        if verbose:
            print('Couldnt find a directory, resulting to default')
        directory = root_directory + 'psy_fits_dev/'
    else:
        directory = root_directory + 'psy_fits_v'+str(version)+'/'
    return directory
 
def process_session(bsid,complete=True,version=None,format_options={},refit=False):
    '''
        Fits the model, dropout analysis, and cross validation
        bsid, behavior_session_id
        complete, if True, does a dropout analysis 
        version, the version of the model, where to save the results. Defaults to "dev"
        format_options, a dictionary of options
    
    '''
    
    # Process directory, filename, and bsid
    if type(bsid) is str:
        bsid = int(bsid)
    directory = get_directory(version, verbose=True)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    filename = directory + str(bsid)
    fig_filename = directory + 'figures_sessions/'+str(bsid)
    print(filename) 

    # Check if this fit has already completed
    if os.path.isfile(filename+".pkl") & (not refit):
        print('Already completed this fit, quitting')
        return

    print('Starting Fit now')
    print("Pulling Data")
    session = pgt.get_data(bsid)

    print("Annotating lick bouts")
    pm.annotate_licks(session) 
    pm.annotate_bouts(session)
   
    print("Formating Data")
    format_options = get_format_options(format_options)
    psydata = format_session(session,format_options)

    print("Initial Fit")
    strategies={'bias','task0','timing1D','omissions','omissions1'}
    if np.sum(session.stimulus_presentations.omitted) == 0:
       strategies.remove('omissions')
       strategies.remove('omissions1')
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,strategies)
    ypred,ypred_each = compute_ypred(psydata, wMode,weights)
    plot_weights(wMode, weights,psydata,errorbar=credibleInt, ypred = ypred,filename=fig_filename)

    print("Cross Validation Analysis")
    cross_psydata = psy.trim(psydata, END=int(np.floor(len(psydata['y'])/format_options['num_cv_folds'])*format_options['num_cv_folds'])) 
    cross_results = compute_cross_validation(cross_psydata, hyp, weights,folds=format_options['num_cv_folds'])
    cv_pred = compute_cross_validation_ypred(cross_psydata, cross_results,ypred)
    
    if complete:
        print("Dropout Analysis")
        models = dropout_analysis(psydata, strategies, format_options)
        #plot_dropout(models,filename=fig_filename)

    print('Packing up and saving')
    try:
        metadata = session.metadata
    except:
        metadata = []
    output = [ hyp,   evd,   wMode,   hess,   credibleInt,   weights,   ypred,  psydata,  cross_results,  cv_pred,  metadata]
    labels = ['hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']       
    fit = dict((x,y) for x,y in zip(labels, output))
    fit['ID'] = bsid

    if complete:
        fit['models'] = models

    if complete:
        fit = cluster_fit(fit,directory=directory) # gets saved separately

    save(filename+".pkl", fit) 
    summarize_fit(fit, version=20, savefig=True)
    plt.close('all')
    
def annotate_stimulus_presentations(session,ignore_trial_errors=False):
    '''
        Adds columns to the stimulus_presentation table describing whether certain task events happened during that flash
        Inputs:
        session, the SDK session object
    
        Appends columns:
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
    session.stimulus_presentations['hits']   =  session.stimulus_presentations['licked'] & session.stimulus_presentations['change']
    session.stimulus_presentations['misses'] = ~session.stimulus_presentations['licked'] & session.stimulus_presentations['change']
    session.stimulus_presentations['aborts'] =  session.stimulus_presentations['licked'] & ~session.stimulus_presentations['change']
    session.stimulus_presentations['in_grace_period'] = (session.stimulus_presentations['time_from_last_change'] <= 4.5) & \
        (session.stimulus_presentations['time_from_last_reward'] <=4.5)
    # Remove Aborts that happened during grace period
    session.stimulus_presentations.at[session.stimulus_presentations['in_grace_period'],'aborts'] = False 

    # These ones require iterating the trials table, and is super slow
    session.stimulus_presentations['false_alarm'] = False
    session.stimulus_presentations['correct_reject'] = False
    session.stimulus_presentations['auto_rewards'] = False
    try:
        for i in session.stimulus_presentations.index:
            found_it=True
            trial = session.trials[
                (session.trials.start_time <= session.stimulus_presentations.at[i,'start_time']) & 
                (session.trials.stop_time >=session.stimulus_presentations.at[i,'start_time'] + 0.25)
                ]
            if len(trial) > 1:
                raise Exception("Could not isolate a trial for this flash")
            if len(trial) == 0:
                trial = session.trials[
                    (session.trials.start_time <= session.stimulus_presentations.at[i,'start_time']) & 
                    (session.trials.stop_time+0.75 >= session.stimulus_presentations.at[i,'start_time'] + 0.25)
                    ]  
                if ( len(trial) == 0 ) & \
                    (session.stimulus_presentations.at[i,'start_time'] > session.trials.start_time.values[-1]):
                    trial = session.trials[session.trials.index == session.trials.index[-1]]
                elif ( len(trial) ==0) & \
                    (session.stimulus_presentations.at[i,'start_time'] < session.trials.start_time.values[0]):
                    trial = session.trials[session.trials.index == session.trials.index[0]]
                elif np.sum(session.trials.aborted) == 0:
                    found_it=False
                elif len(trial) == 0:
                    trial = session.trials[
                        (session.trials.start_time <= session.stimulus_presentations.at[i,'start_time']+0.75) & 
                        (session.trials.stop_time+0.75 >= session.stimulus_presentations.at[i,'start_time'] + 0.25)
                        ]  
                    if len(trial) == 0: 
                        print('stim index: '+str(i))
                        raise Exception("Could not find a trial for this flash")
            if found_it:
                if trial['false_alarm'].values[0]:
                    if (trial.change_time.values[0] >= session.stimulus_presentations.at[i,'start_time']) & \
                        (trial.change_time.values[0] <= session.stimulus_presentations.at[i,'stop_time'] ):
                        session.stimulus_presentations.at[i,'false_alarm'] = True
                if trial['correct_reject'].values[0]:
                    if (trial.change_time.values[0] >= session.stimulus_presentations.at[i,'start_time']) & \
                        (trial.change_time.values[0] <= session.stimulus_presentations.at[i,'stop_time'] ):
                        session.stimulus_presentations.at[i,'correct_reject'] = True
                if trial['auto_rewarded'].values[0]:
                    if (trial.change_time.values[0] >= session.stimulus_presentations.at[i,'start_time']) & \
                        (trial.change_time.values[0] <= session.stimulus_presentations.at[i,'stop_time'] ):
                        session.stimulus_presentations.at[i,'auto_rewards'] = True
    except:
        if ignore_trial_errors:
            print('WARNING, had trial alignment errors, but are ignoring due to ignore_trial_errors=True')
        else:
            raise Exception('Trial Alignment Error. Set ignore_trial_errors=True to ignore. Flash #: '+str(i))

def get_format_options(format_options):
    '''
        Defines the default format options, and sets any values not passed in
    '''
    defaults = {'fit_bouts':True,
                'timing0/1':True,
                'mean_center':False,
                'timing_params':np.array([-5,4]),
                'timing_params_session':np.array([-5,4]),
                'ignore_trial_errors':False,
                'num_cv_folds':10
                }
    for k in defaults.keys():
        if k not in format_options:
            format_options[k] = defaults[k]

    return format_options

def format_session(session,format_options):
    '''
        Formats the data into the requirements of Psytrack
        ARGS:
            data outputed from SDK
            format_options, a dictionary with keys:
                fit_bouts, if True (Default), then fits to the start of each licking bout, instead of each lick
                timing0/1, if True (Default), then timing is a vector of 0s and 1s, otherwise, is -1/+1
                mean_center, if True, then regressors are mean-centered
                timing_params, [p1,p2] parameters for 1D timing regressor
                timing_params_session, parameters custom fit for this session
                                
        Returns:
            data formated for psytrack. A dictionary with key/values:
            psydata['y'] = a vector of no-licks (1) and licks(2) for each flashes
            psydata['inputs'] = a dictionary with each key an input ('random','timing', 'task', etc)
                each value has a 2D array of shape (N,M), where N is number of flashes, and M is 1 unless you want to look at history/flash interaction terms
    '''     
    if len(session.licks) < 10:
        raise Exception('Less than 10 licks in this session')   

    # Build Dataframe of flashes
    annotate_stimulus_presentations(session,ignore_trial_errors = format_options['ignore_trial_errors'])
    df = pd.DataFrame(data = session.stimulus_presentations.start_time)
    if format_options['fit_bouts']:
        lick_bouts = session.stimulus_presentations.bout_start.values
        df['y'] = np.array([2 if x else 1 for x in lick_bouts])
    else:
        df['y'] = np.array([2 if x else 1 for x in session.stimulus_presentations.licked])
    df['hits']          = session.stimulus_presentations.hits
    df['misses']        = session.stimulus_presentations.misses
    df['false_alarm']   = session.stimulus_presentations.false_alarm
    df['correct_reject']= session.stimulus_presentations.correct_reject
    df['aborts']        = session.stimulus_presentations.aborts
    df['auto_rewards']  = session.stimulus_presentations.auto_rewards
    df['start_time']    = session.stimulus_presentations.start_time
    df['change']        = session.stimulus_presentations.change
    df['omitted']       = session.stimulus_presentations.omitted  
    df['licked']        = session.stimulus_presentations.licked
    df['included']      = True
 
    # Build Dataframe of regressors
    if format_options['fit_bouts']:
        df['bout_start']        = session.stimulus_presentations['bout_start']
        df['bout_end']          = session.stimulus_presentations['bout_end']
        df['num_bout_start']    = session.stimulus_presentations['num_bout_start']
        df['num_bout_end']      = session.stimulus_presentations['num_bout_end']
        df['flashes_since_last_lick'] = session.stimulus_presentations.groupby(session.stimulus_presentations['bout_end'].cumsum()).cumcount(ascending=True)
        df['in_bout_raw_bad']   = session.stimulus_presentations['bout_start'].cumsum() > session.stimulus_presentations['bout_end'].cumsum()
        df['in_bout_raw']       = session.stimulus_presentations['num_bout_start'].cumsum() > session.stimulus_presentations['num_bout_end'].cumsum()
        df['in_bout']           = np.array([1 if x else 0 for x in df['in_bout_raw'].shift(fill_value=False)])
        df['task0']             = np.array([1 if x else 0 for x in df['change']])
        df['task1']             = np.array([1 if x else -1 for x in df['change']])
        df['late_task0']        = df['task0'].shift(1,fill_value=0)
        df['late_task1']        = df['task1'].shift(1,fill_value=-1)
        df['taskCR']            = np.array([0 if x else -1 for x in df['change']])
        df['omissions']         = np.array([1 if x else 0 for x in df['omitted']])
        df['omissions1']        = np.array([x for x in np.concatenate([[0], df['omissions'].values[0:-1]])])
        df['timing1D']          = np.array([timing_sigmoid(x,format_options['timing_params']) for x in df['flashes_since_last_lick'].shift()])
        df['timing1D_session']  = np.array([timing_sigmoid(x,format_options['timing_params_session']) for x in df['flashes_since_last_lick'].shift()])
        if format_options['timing0/1']:
            min_timing_val = 0
        else:
            min_timing_val = -1
        df['timing1'] =  np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==0])
        df['timing2'] =  np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==1])
        df['timing3'] =  np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==2])
        df['timing4'] =  np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==3])
        df['timing5'] =  np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==4])
        df['timing6'] =  np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==5])
        df['timing7'] =  np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==6])
        df['timing8'] =  np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==7])
        df['timing9'] =  np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==8])
        df['timing10'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() ==9])
        df['included'] = df['in_bout'] ==0
        full_df = copy.copy(df)
        df = df[df['in_bout']==0] 
        df['missing_trials'] = np.concatenate([np.diff(df.index)-1,[0]])
    else:
        # Fit each lick, not lick bouts
        df['task0']      = np.array([1 if x else 0 for x in df['change']])
        df['task1']      = np.array([1 if x else -1 for x in df['change']])
        df['taskCR']     = np.array([0 if x else -1 for x in df['change']])
        df['omissions']  = np.array([1 if x else 0 for x in df['omitted']])
        df['omissions1'] = np.concatenate([[0], df['omissions'].values[0:-1]])
        df['flashes_since_last_lick'] = df.groupby(df['licked'].cumsum()).cumcount(ascending=True)
        if format_options['timing0/1']:
           min_timing_val = 0
        else:
           min_timing_val = -1   
        df['timing2'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() >=2])
        df['timing3'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() >=3])
        df['timing4'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() >=4])
        df['timing5'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() >=5])
        df['timing6'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() >=6])
        df['timing7'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() >=7])
        df['timing8'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() >=8])
        df['timing9'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() >=9]) 
        df['timing10'] = np.array([1 if x else min_timing_val for x in df['flashes_since_last_lick'].shift() >=10]) 
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
                'timing1D': df['timing1D'].values[:,np.newaxis],
                'timing1D_session': df['timing1D_session'].values[:,np.newaxis],
                'late_task1':df['late_task1'].values[:,np.newaxis],
                'late_task0':df['late_task0'].values[:,np.newaxis]} 
   
    # Mean Center the regressors should you need to 
    if format_options['mean_center']:
        for key in inputDict.keys():
            # mean center
            inputDict[key] = inputDict[key] - np.mean(inputDict[key])
   
    # After Mean centering, include missing trials
    inputDict['missing_trials'] = df['missing_trials'].values[:,np.newaxis]

    # Pack up data into format for psytrack
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

def timing_sigmoid(x,params,min_val = -1, max_val = 0,tol=1e-3):
    '''
        Evaluates a sigmoid between min_val and max_val with parameters params
    '''
    if np.isnan(x):
        x = 0
    x_corrected = x+1
    y = min_val+(max_val-min_val)/(1+(x_corrected/params[1])**params[0])
    if (y -min_val) < tol:
        y = min_val
    if (max_val - y) < tol:
        y = max_val
    return y
   
def fit_weights(psydata, strategies, fit_overnight=False):
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
    if 'bias' in strategies:      weights['bias'] = 1
    if 'task0' in strategies:     weights['task0'] = 1
    if 'task1' in strategies:     weights['task1'] = 1
    if 'taskcr' in strategies:    weights['taskcr'] = 1
    if 'omissions' in strategies: weights['omissions'] = 1
    if 'omissions1' in strategies:weights['omissions1'] = 1
    if 'timing1' in strategies:   weights['timing1'] = 1
    if 'timing2' in strategies:   weights['timing2'] = 1
    if 'timing3' in strategies:   weights['timing3'] = 1
    if 'timing4' in strategies:   weights['timing4'] = 1
    if 'timing5' in strategies:   weights['timing5'] = 1
    if 'timing6' in strategies:   weights['timing6'] = 1
    if 'timing7' in strategies:   weights['timing7'] = 1
    if 'timing8' in strategies:   weights['timing8'] = 1
    if 'timing9' in strategies:   weights['timing9'] = 1
    if 'timing10' in strategies:  weights['timing10'] = 1
    if 'timing1D' in strategies:  weights['timing1D'] = 1
    if 'timing1D_session' in strategies: weights['timing1D_session'] = 1
    if 'late_task' in strategies: weights['late_task0'] = 1
    print(weights)

    K = np.sum([weights[i] for i in weights.keys()])
    hyper = {'sigInit': 2**4.,
            'sigma':[2**-4.]*K,
            'sigDay': 2**4}
    if fit_overnight:
        optList=['sigma','sigDay']
    else:
        optList=['sigma']
    hyp,evd,wMode,hess =psy.hyperOpt(psydata,hyper,weights, optList)
    credibleInt = hess['W_std']
    return hyp, evd, wMode, hess, credibleInt, weights

def compute_ypred(psydata, wMode, weights):
    '''
        Makes a full model prediction from the wMode
        Returns:
        pR, the probability of licking on each image
        pR_each, the contribution of licking from each weight. These contributions 
            interact nonlinearly, so this is an approximation. 
    '''
    g = psy.read_input(psydata, weights)
    gw = g*wMode.T
    total_gw = np.sum(gw,axis=1)
    pR = transform(total_gw)
    pR_each = transform(gw) 
    return pR, pR_each

def transform(series):
    '''
        passes the series through the logistic function
    '''
    return 1/(1+np.exp(-(series)))

def get_weights_list(weights):
    '''
        Return a sorted list of the weights in the model
    '''
    weights_list = []
    for i in sorted(weights.keys()):
        weights_list += [i]*weights[i]
    return weights_list

def clean_weights(weights):
    '''
        Return a cleaned up list of weights suitable for plotting labels
    '''
    weight_dict = {
    'bias':'Bias',
    'omissions':'Omitted',
    'omissions0':'Omitted',
    'omissions1':'Prev. Omitted',
    'task0':'Visual',
    'timing1D':'Timing'}

    clean_weights = []
    for w in weights:
        if w in weight_dict.keys():
            clean_weights.append(weight_dict[w])
        else:
            clean_weights.append(w)
    return clean_weights

def clean_dropout(weights):
    '''
        Return a cleaned up list of dropouts suitable for plotting labels 
    '''
    weight_dict = {
    'Bias':'Bias',
    'Omissions':'Omitted',
    'Omissions1':'Prev. Omitted',
    'Task0':'Visual',
    'timing1D':'Timing',
    'Full-Task0':'Full Model'}

    clean_weights = []
    for w in weights:
        if w in weight_dict.keys():
            clean_weights.append(weight_dict[w])
        else:
            clean_weights.append(w)
    return clean_weights

def plot_weights(wMode,weights,psydata,errorbar=None, ypred=None,START=0, END=0,plot_trials=True,session_labels=None, seedW = None,ypred_each = None,filename=None,cluster_labels=None,smoothing_size=50,num_clusters=None):
    '''
        Plots the fit results by plotting the weights in linear and probability space. 
        wMode, the weights
        weights, the dictionary of strategyes
        psydata, the dataset
        errorbar, the std of the weights
        ypred, the full model prediction
        START, the flash number to start on
        END, the flash number to end on
     
    '''
    # Determine x axis limits
    K,N = wMode.shape    
    if START <0: START = 0
    if START > N: raise Exception(" START > N")
    if END <=0: END = N
    if END > N: END = N
    if START >= END: raise Exception("START >= END")

    # initialize 
    weights_list = clean_weights(get_weights_list(weights))
    my_colors = sns.color_palette("hls",len(weights.keys()))
    if 'dayLength' in psydata:
        dayLength = np.concatenate([[0],np.cumsum(psydata['dayLength'])])
    else:
        dayLength = []

    # Determine which panels to plot
    if (ypred is not None) & plot_trials:
        fig,ax = plt.subplots(nrows=4,ncols=1, figsize=(10,10))
        trial_ax = 2
        full_ax = 3
        cluster_ax = 3
    elif plot_trials:
        fig,ax = plt.subplots(nrows=3,ncols=1, figsize=(10,8))  
        cluster_ax = 2
        trial_ax = 2
    elif (ypred is not None):
        fig,ax = plt.subplots(nrows=3,ncols=1, figsize=(10,8))
        cluster_ax = 2
        full_ax = 2
    else:
        fig,ax = plt.subplots(nrows=2,ncols=1, figsize=(10,6))
        cluster_ax = 1

    # Axis 0, plot weights
    for i in np.arange(0, len(weights_list)):
        ax[0].plot(wMode[i,:], linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])        
        ax[0].fill_between(np.arange(len(wMode[i])), wMode[i,:]-2*errorbar[i], 
            wMode[i,:]+2*errorbar[i],facecolor=my_colors[i], alpha=0.1)    
        if seedW is not None:
            ax[0].plot(seedW[i,:], linestyle="--", lw=2, color=my_colors[i], label= "seed "+weights_list[i])
    ax[0].plot([0,np.shape(wMode)[1]], [0,0], 'k--',alpha=0.2)
    ax[0].set_ylabel('Weight',fontsize=12)
    ax[0].set_xlabel('Flash #',fontsize=12)
    ax[0].set_xlim(START,END)
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].tick_params(axis='both',labelsize=12)
    for i in np.arange(0, len(dayLength)-1):
        ax[0].axvline(dayLength[i],color='k',alpha=0.2)
        if session_labels is not None:
            ax[0].text(dayLength[i],ax[0].get_ylim()[1], session_labels[i],rotation=25)

    # Axis 1, plot nonlinear weights (approximation)
    for i in np.arange(0, len(weights_list)):
        ax[1].plot(transform(wMode[i,:]), linestyle="-", lw=3, color=my_colors[i],label=weights_list[i])
        ax[1].fill_between(np.arange(len(wMode[i])), transform(wMode[i,:]-2*errorbar[i]), 
            transform(wMode[i,:]+2*errorbar[i]),facecolor=my_colors[i], alpha=0.1)                  
        if seedW is not None:
            ax[1].plot(transform(seedW[i,:]), linestyle="--", lw=2, color=my_colors[i], label= "seed "+weights_list[i])
    ax[1].set_ylim(0,1)
    ax[1].set_ylabel('Lick Prob',fontsize=12)
    ax[1].set_xlabel('Flash #',fontsize=12)
    ax[1].set_xlim(START,END)
    ax[1].tick_params(axis='both',labelsize=12)
    for i in np.arange(0, len(dayLength)-1):
        ax[1].plot([dayLength[i], dayLength[i]],[0,1], 'k-',alpha=0.2)

    # scatter plot of trials
    if plot_trials:
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
        ax[2].set_yticklabels(['hits','miss','CR','FA','abort','auto'],fontdict={'fontsize':12})
        ax[2].set_xlim(START,END)
        ax[2].set_xlabel('Flash #',fontsize=12)
        ax[2].tick_params(axis='both',labelsize=12)

    # Plot Full Model prediction and comparison with data
    if (ypred is not None):
        ax[full_ax].plot(pgt.moving_mean(ypred,smoothing_size), 'k',alpha=0.3,label='Full Model (n='+str(smoothing_size)+ ')')
        if ypred_each is not None:
            for i in np.arange(0, len(weights_list)):
                ax[full_ax].plot(ypred_each[:,i], linestyle="-", lw=3, alpha = 0.3,color=my_colors[i],label=weights_list[i])        
        ax[full_ax].plot(pgt.moving_mean(psydata['y']-1,smoothing_size), 'b',alpha=0.5,label='data (n='+str(smoothing_size)+ ')')
        ax[full_ax].set_ylim(0,1)
        ax[full_ax].set_ylabel('Lick Prob',fontsize=12)
        ax[full_ax].set_xlabel('Flash #',fontsize=12)
        ax[full_ax].set_xlim(START,END)
        ax[full_ax].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[full_ax].tick_params(axis='both',labelsize=12)

    # plot session clustering
    if cluster_labels is not None:
        cp = np.where(~(np.diff(cluster_labels) == 0))[0]
        cp = np.concatenate([[0], cp, [len(cluster_labels)]])
        if num_clusters is None:
            num_clusters = len(np.unique(cluster_labels))
        cluster_colors = sns.color_palette("hls",num_clusters)
        for i in range(0, len(cp)-1):
            ax[cluster_ax].axvspan(cp[i],cp[i+1],color=cluster_colors[cluster_labels[cp[i]+1]], alpha=0.3)
    
    # Save
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename+"_weights.png")
   
def compute_cross_validation(psydata, hyp, weights,folds=10):
    '''
        Computes Cross Validation for the data given the regressors as defined in hyp and weights
    '''
    trainDs, testDs = split_data(psydata,F=folds)
    test_results = []
    for k in range(folds):
        print("\rrunning fold " +str(k),end="")
        _,_,wMode_K,_ = psy.hyperOpt(trainDs[k], hyp, weights, ['sigma'],hess_calc=None)
        logli, gw = xval_loglike(testDs[k], wMode_K, trainDs[k]['missing_trials'], weights)
        res = {'logli' : np.sum(logli), 'gw' : gw, 'test_inds' : testDs[k]['test_inds']}
        test_results += [res]
   
    print("") 
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
    cv_pred = 1/(1+np.exp(-X))

    # Fill in untested indicies with ypred, these come from end
    full_pred = copy.copy(ypred)
    full_pred[np.where(xval_mask==True)[0]] = cv_pred
    return full_pred
  
def dropout_analysis(psydata, strategies,format_options):
    '''
        Computes a dropout analysis for the data in psydata. In general, computes a full set, and then removes each feature one by one. Also computes hard-coded combinations of features
        Returns a list of models and a list of labels for each dropout
    '''
    models =dict()

    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,strategies)
    cross_psydata = psy.trim(psydata, END=int(np.floor(len(psydata['y'])/format_options['num_cv_folds'])*format_options['num_cv_folds'])) 
    cross_results = compute_cross_validation(cross_psydata, hyp, weights,folds=format_options['num_cv_folds'])
    models['Full'] = (hyp, evd, wMode, hess, credibleInt,weights,cross_results)

    # Iterate through strategies and remove them
    for s in strategies:
        dropout_strategies = copy.copy(strategies)
        dropout_strategies.remove(s)
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,dropout_strategies)
        cross_results = compute_cross_validation(cross_psydata, hyp, weights,folds=format_options['num_cv_folds'])
        models[s] = (hyp, evd, wMode, hess, credibleInt,weights,cross_results)

    return models

def plot_dropout(models,filename=None):
    '''
        Plots the dropout results for a single session
        
    '''
    plt.figure(figsize=(10,3.5))
    ax = plt.gca()
    labels = sorted(list(models.keys()))
    labels.remove('Full')
    for i,m in enumerate(labels):
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
        plt.plot(i, (1-models[m][1]/models['Full'][1])*100, 'ko')
    plt.xlabel('Model Component',fontsize=12)
    plt.ylabel('% change in evidence',fontsize=12)
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xticks(np.arange(0,len(labels)))
    ax.set_xticklabels(labels,rotation=90)
    plt.tight_layout()
    ax.axhline(0,color='k',alpha=0.2)
    plt.ylim(ymax=5,ymin=-20)
    if filename is not None:
        plt.savefig(filename+"_dropout.png")

def plot_session_summary_priors(IDS,version=None,savefig=False,group_label="",fs1=12,fs2=12,filetype='.png'):
    '''
        Make a summary plot of the priors on each feature
    '''
    directory=get_directory(version)

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    alld = []
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass 
        else:
            sigmas = session_summary[0]
            weights = session_summary[1]
            ax.plot(np.arange(0,len(sigmas)),sigmas, 'o',alpha = 0.5)
            plt.yscale('log')
            plt.ylim(0.0001, 20)
            ax.set_xticks(np.arange(0,len(sigmas)))
            weights_list = clean_weights(get_weights_list(weights))
            ax.set_xticklabels(weights_list,fontsize=fs2,rotation=90)
            plt.ylabel('Smoothing Prior, $\sigma$\n <-- smooth           variable --> ',fontsize=fs1)
            counter +=1
            alld.append(sigmas)            

    if counter == 0:
        print('NO DATA')
        return
    alld = np.mean(np.vstack(alld),0)
    for i in np.arange(0, len(sigmas)):
        ax.plot([i-.25, i+.25],[alld[i],alld[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    ax.axhline(0.001,color='k',alpha=0.2)
    ax.axhline(0.01,color='k',alpha=0.2)
    ax.axhline(0.1,color='k',alpha=0.2)
    ax.axhline(1,color='k',alpha=0.2)
    ax.axhline(10,color='k',alpha=0.2)
    plt.yticks(fontsize=fs2-4,rotation=90)
    ax.xaxis.tick_top()
    ax.set_xlim(xmin=-.5)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"prior"+filetype)

def plot_session_summary_correlation(IDS,version=None,savefig=False,group_label="",verbose=True):
    '''
        Make a summary plot of the priors on each feature
    '''
    directory=get_directory(version)
    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    scores = []
    ids = []
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
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
        plt.savefig(directory+"figures_summary/summary_"+group_label+"correlation.png")
    if verbose:
        median = np.argsort(np.array(scores))[len(scores)//2]
        best = np.argmax(np.array(scores))
        worst = np.argmin(np.array(scores)) 
        print('R^2 Correlation:')
        print('Worst  Session: ' + str(ids[worst]) + " " + str(scores[worst]))
        print('Median Session: ' + str(ids[median]) + " " + str(scores[median]))
        print('Best   Session: ' + str(ids[best]) + " " + str(scores[best]))      
    return scores, ids 

def plot_session_summary_dropout(IDS,version=None,cross_validation=True,savefig=False,group_label="",model_evidence=False,fs1=12,fs2=12,filetype='.png'):
    '''
        Make a summary plot showing the fractional change in either model evidence (not cross-validated), or log-likelihood (cross-validated)
    '''
    directory=get_directory(version)
    
    # make figure    
    fig,ax = plt.subplots(figsize=(7.2,6))
    alld = []
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    if cross_validation:
        plt.ylabel('% Change in CV Likelihood \n <-- Worse Fit',fontsize=fs1)
    else:
        plt.ylabel('% Change in Likelihood \n <-- Worse Fit',fontsize=fs1)

    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version, cross_validation_dropout=cross_validation)
        except:
            pass
        else:
            dropout_dict = session_summary[2]
            labels  = session_summary[3]
            dropout = [dropout_dict[x] for x in labels[1:]]
            ax.plot(np.arange(0,len(dropout)),dropout, 'o',alpha=0.5)
            alld.append(dropout)
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    alld = np.mean(np.vstack(alld),0)
    plt.yticks(fontsize=fs2-4,rotation=90)
    for i in np.arange(0, len(dropout)):
        ax.plot([i-.25, i+.25],[alld[i],alld[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    ax.set_xticks(np.arange(0,len(dropout)))
    ax.set_xticklabels(clean_weights(labels[1:]),fontsize=fs2, rotation = 90)
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.xlim(-0.5,len(dropout) - 0.5)
    plt.ylim(-80,5)
    if savefig:
        if cross_validation:
            plt.savefig(directory+"figures_summary/summary_"+group_label+"dropout_cv"+filetype)
        else:
            plt.savefig(directory+"figures_summary/summary_"+group_label+"dropout"+filetype)

def plot_session_summary_weights(IDS,version=None, savefig=False,group_label="",return_weights=False,fs1=12,fs2=12,filetype='.svg',hit_threshold=0):
    '''
        Makes a summary plot showing the average weight value for each session
    '''
    directory=get_directory(version)

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    all_weights = []
    good = []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version,hit_threshold=hit_threshold)
        except:
            good.append(False)
        else:
            good.append(True)
            avgW = session_summary[4]
            weights  = session_summary[1]
            ax.plot(np.arange(0,len(avgW)),avgW, 'o',alpha=0.5)
            ax.set_xticks(np.arange(0,len(avgW)))
            plt.ylabel('Avg. Weights across each session',fontsize=fs1)

            all_weights.append(avgW)
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    allW = np.mean(np.vstack(all_weights),0)
    for i in np.arange(0, len(avgW)):
        ax.plot([i-.25, i+.25],[allW[i],allW[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    weights_list = get_weights_list(weights)
    ax.set_xticklabels(clean_weights(weights_list),fontsize=fs2, rotation = 90)
    ax.xaxis.tick_top()
    plt.yticks(fontsize=fs2-4,rotation=90)
    plt.tight_layout()
    plt.xlim(-0.5,len(avgW) - 0.5)
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weights"+filetype)
    if return_weights:
        return all_weights, good

def plot_session_summary_weight_range(IDS,version=None,savefig=False,group_label=""):
    '''
        Makes a summary plot showing the range of each weight across each session
    '''
    directory=get_directory(version)

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    allW = None
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    all_range = []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            rangeW = session_summary[5]
            weights  = session_summary[1]
            ax.plot(np.arange(0,len(rangeW)),rangeW, 'o',alpha=0.5)
            ax.set_xticks(np.arange(0,len(rangeW)))
            plt.ylabel('Range of Weights across each session',fontsize=12)
            all_range.append(rangeW)    
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    allW = np.mean(np.vstack(all_range),0)
    for i in np.arange(0, len(rangeW)):
        ax.plot([i-.25, i+.25],[allW[i],allW[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    weights_list = get_weights_list(weights)
    ax.set_xticklabels(clean_weights(weights_list),fontsize=12, rotation = 90)
    ax.xaxis.tick_top()
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.xlim(-0.5,len(rangeW) - 0.5)
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_range.png")

def plot_session_summary_weight_scatter(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against each other weight, plotting the average weight for each session
    '''
    directory = get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=nel,ncols=nel,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = get_weights_list(weights)
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
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_scatter.png")

def plot_session_summary_dropout_scatter(IDS,version=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of the dropout performance change for each feature against each other feature 
    '''
    directory=get_directory(version)
    # make figure    
    allW = None
    counter = 0
    first = True
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version, cross_validation_dropout=True)
        except:
            pass
        else:
            if first:
                fig,ax = plt.subplots(nrows=len(session_summary[2])-1,ncols=len(session_summary[2])-1,figsize=(11,10))        
                first = False 
            d = session_summary[2]
            l = session_summary[3]
            dropout = d
            labels = l
            dropout= [d[x] for x in labels[1:]]
            labels = labels[1:]
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
                    ax[i,j-1].set_xlabel(clean_dropout([labels[j]])[0],fontsize=12)
                    ax[i,j-1].set_ylabel(clean_dropout([labels[i]])[0],fontsize=12)
                    ax[i,j-1].xaxis.set_tick_params(labelsize=12)
                    ax[i,j-1].yaxis.set_tick_params(labelsize=12)
                    if i == 0:
                        ax[i,j-1].set_ylim(-80,5)
            counter+=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"dropout_scatter.png")

def plot_session_summary_weight_avg_scatter(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against each other weight, plotting the average weight for each session
    '''
    directory = get_directory(version)

    # make figure    
    fig,ax = plt.subplots(nrows=nel,ncols=nel,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = get_weights_list(weights)
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
                    ax[i,j-1].set_xlabel(clean_weights([weights_list[j]])[0],fontsize=12)
                    ax[i,j-1].set_ylabel(clean_weights([weights_list[i]])[0],fontsize=12)
                    ax[i,j-1].xaxis.set_tick_params(labelsize=12)
                    ax[i,j-1].yaxis.set_tick_params(labelsize=12)
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter.png")

# UPDATE_REQUIRED
def plot_session_summary_weight_avg_scatter_1_2(IDS,label1='late_task0',label2='timing1D',directory=None,savefig=False,group_label="",nel=3,fs1=12,fs2=12,filetype='.png',plot_error=True):
    '''
        Makes a summary plot of the average weights of task0 against omission weights for each session
        Also computes a regression line, and returns the linear model
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(3,4))
    allx = []
    ally = []
    counter = 0
    ax.axvline(0,color='k',alpha=0.5,ls='--')
    ax.axhline(0,color='k',alpha=0.5,ls='--')
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = get_weights_list(weights)
            xdex = np.where(np.array(weights_list) == label1)[0][0]
            ydex = np.where(np.array(weights_list) == label2)[0][0]

            meanWj = np.mean(W[xdex,:])
            meanWi = np.mean(W[ydex,:])
            allx.append(meanWj)
            ally.append(meanWi)
            stdWj = np.std(W[xdex,:])
            stdWi = np.std(W[ydex,:])
            if plot_error:
                ax.plot([meanWj, meanWj], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax.plot(meanWj+[-stdWj,stdWj], [meanWi, meanWi],'k-',alpha=0.1)
            ax.plot(meanWj, meanWi,'ko',alpha=0.5)
            ax.set_xlabel(clean_weights([weights_list[xdex]])[0],fontsize=fs1)
            ax.set_ylabel(clean_weights([weights_list[ydex]])[0],fontsize=fs1)
            ax.xaxis.set_tick_params(labelsize=fs2)
            ax.yaxis.set_tick_params(labelsize=fs2)
            counter+=1
    if counter == 0:
        print('NO DATA')
        return
    x = np.array(allx).reshape((-1,1))
    y = np.array(ally)
    model = LinearRegression(fit_intercept=True).fit(x,y)
    sortx = np.sort(allx).reshape((-1,1))
    y_pred = model.predict(sortx)
    ax.plot(sortx,y_pred, 'r--')
    score = round(model.score(x,y),2)
    #plt.text(sortx[0],y_pred[-1],"Omissions = "+str(round(model.coef_[0],2))+"*Task \nr^2 = "+str(score),color="r",fontsize=fs2)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_"+label1+"_"+label2+filetype)
    return model


def plot_session_summary_weight_avg_scatter_task0(IDS,version=None,savefig=False,group_label="",nel=3,fs1=12,fs2=12,filetype='.png',plot_error=True):
    '''
        Makes a summary plot of the average weights of task0 against omission weights for each session
        Also computes a regression line, and returns the linear model
    '''
    directory=get_directory(version) 
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(3,4))
    allx = []
    ally = []
    counter = 0
    ax.axvline(0,color='k',alpha=0.5,ls='--')
    ax.axhline(0,color='k',alpha=0.5,ls='--')
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = get_weights_list(weights)
            xdex = np.where(np.array(weights_list) == 'task0')[0][0]
            ydex = np.where(np.array(weights_list) == 'omissions1')[0][0]

            meanWj = np.mean(W[xdex,:])
            meanWi = np.mean(W[ydex,:])
            allx.append(meanWj)
            ally.append(meanWi)
            stdWj = np.std(W[xdex,:])
            stdWi = np.std(W[ydex,:])
            if plot_error:
                ax.plot([meanWj, meanWj], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax.plot(meanWj+[-stdWj,stdWj], [meanWi, meanWi],'k-',alpha=0.1)
            ax.plot(meanWj, meanWi,'ko',alpha=0.5)
            ax.set_xlabel('Avg. '+clean_weights([weights_list[xdex]])[0]+' weight',fontsize=fs1)
            ax.set_ylabel('Avg. '+clean_weights([weights_list[ydex]])[0]+' weight',fontsize=fs1)
            ax.xaxis.set_tick_params(labelsize=fs2)
            ax.yaxis.set_tick_params(labelsize=fs2)
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
    #plt.text(sortx[0],y_pred[-1],"Omissions = "+str(round(model.coef_[0],2))+"*Task \nr^2 = "+str(score),color="r",fontsize=fs2)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_task0"+filetype)
    return model

def plot_session_summary_weight_avg_scatter_hits(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against the total number of hits
    '''
    directory=get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=2,ncols=nel+1,figsize=(14,6))
    allW = None
    counter = 0
    xmax = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            fit = session_summary[7]
            hits = np.sum(fit['psydata']['hits'])
            xmax = np.max([hits, xmax])
            weights  = session_summary[1]
            weights_list = get_weights_list(weights)
            for i in np.arange(0,np.shape(W)[0]):
                ax[0,i].axhline(0,color='k',alpha=0.1)
                meanWi = np.mean(W[i,:])
                stdWi = np.std(W[i,:])
                ax[0,i].plot([hits, hits], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax[0,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[0,i].set_xlabel('hits',fontsize=12)
                ax[0,i].set_ylabel(clean_weights([weights_list[i]])[0],fontsize=12)
                ax[0,i].xaxis.set_tick_params(labelsize=12)
                ax[0,i].yaxis.set_tick_params(labelsize=12)
                ax[0,i].set_xlim(xmin=0,xmax=xmax)

                meanWi = transform(np.mean(W[i,:]))
                stdWiPlus = transform(np.mean(W[i,:])+np.std(W[i,:]))
                stdWiMinus =transform(np.mean(W[i,:])-np.std(W[i,:])) 
                ax[1,i].plot([hits, hits], [stdWiMinus, stdWiPlus],'k-',alpha=0.1)
                ax[1,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[1,i].set_xlabel('hits',fontsize=12)
                ax[1,i].set_ylabel(clean_weights([weights_list[i]])[0],fontsize=12)
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
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_hits.png")

def plot_session_summary_weight_avg_scatter_false_alarms(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against the total number of false_alarms
    '''
    directory = get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=2,ncols=nel+1,figsize=(14,6))
    allW = None
    counter = 0
    xmax = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            fit = session_summary[7]
            hits = np.sum(fit['psydata']['false_alarms'])
            xmax = np.max([hits, xmax])
            weights  = session_summary[1]
            weights_list = clean_weights(get_weights_list(weights))
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
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_false_alarms.png")

def plot_session_summary_weight_avg_scatter_miss(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against the total number of miss
    '''
    directory=get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=2,ncols=nel+1,figsize=(14,6))
    allW = None
    counter = 0
    xmax = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            fit = session_summary[7]
            hits = np.sum(fit['psydata']['misses'])
            xmax = np.max([hits, xmax])
            weights  = session_summary[1]
            weights_list = clean_weights(get_weights_list(weights))
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
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_misses.png")

def plot_session_summary_weight_trajectory(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a summary plot by plotting each weights trajectory across each session. Plots the average trajectory in bold
        this function is super hacky. average is wrong, and doesnt properly align time due to consumption bouts. But gets the general pictures. 
    '''
    directory= get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=nel+1,ncols=1,figsize=(6,10))
    allW = []
    counter = 0
    xmax  =  []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = clean_weights(get_weights_list(weights))
            for i in np.arange(0,np.shape(W)[0]):
                ax[i].plot(W[i,:],alpha = 0.2)
                ax[i].set_ylabel(weights_list[i],fontsize=12)

                xmax.append(len(W[i,:]))
                ax[i].set_xlim(0,np.max(xmax))
                ax[i].xaxis.set_tick_params(labelsize=12)
                ax[i].yaxis.set_tick_params(labelsize=12)
                if i == np.shape(W)[0] -1:
                    ax[i].set_xlabel('Flash #',fontsize=12)
            W = np.pad(W,([0,0],[0,4000]),'constant',constant_values=0)
            allW.append(W[:,0:4000])
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    allW = np.mean(np.array(allW),0)
    for i in np.arange(0,np.shape(W)[0]):
        ax[i].axhline(0, color='k')
        ax[i].plot(allW[i,:],'k',alpha = 1,lw=3)
        if i> 0:
            ax[i].set_ylim(ymin=-2.5)
        ax[i].set_xlim(0,4000)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_trajectory.png")

def get_cross_validation_dropout(cv_results):
    '''
        computes the full log likelihood by summing each cross validation fold
    '''
    return np.sum([i['logli'] for i in cv_results]) 

          
def get_session_summary(behavior_session_id,cross_validation_dropout=True,model_evidence=False,version=None,hit_threshold=0):
    '''
        Extracts useful summary information about each fit
        if cross_validation_dropout, then uses the dropout analysis where each reduced model is cross-validated
    '''
    directory = get_directory(version)
    fit = load_fit(behavior_session_id, version=version)

    if type(fit) is not dict:
        labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
        fit = dict((x,y) for x,y in zip(labels, fit))

    if np.sum(fit['psydata']['hits']) < hit_threshold:
        raise Exception('Below hit threshold')    

    # compute statistics
    dropout = get_session_dropout(fit,cross_validation=cross_validation_dropout)
    avgW = np.mean(fit['wMode'],1)
    rangeW = np.ptp(fit['wMode'],1)
    labels =sorted(list(fit['models'].keys()))
    return fit['hyp']['sigma'],fit['weights'],dropout,labels, avgW, rangeW,fit['wMode'],fit

def plot_session_summary(IDS,version=None,savefig=False,group_label="",nel=4):
    '''
        Makes a series of summary plots for all the IDS
    '''
    directory=get_directory(version)
    plot_session_summary_priors(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout(IDS,version=version,cross_validation=False,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout(IDS,version=version,cross_validation=True,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout_scatter(IDS, version=version, savefig=savefig, group_label=group_label); plt.close('all')
    plot_session_summary_weights(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_weight_range(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_weight_scatter(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter_task0(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter_hits(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter_miss(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter_false_alarms(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_trajectory(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_logodds(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_correlation(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_roc(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_static_comparison(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')

def plot_session_summary_logodds(IDS,version=None,savefig=False,group_label="",cross_validation=True,hit_threshold=0):
    '''
        Makes a summary plot of the log-odds of the model fits = log(prob(lick|lick happened)/prob(lick|no lick happened))
    '''
    directory=get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4.5))
    logodds=[]
    counter =0
    ids= []
    for id in IDS:
        try:
            #session_summary = get_session_summary(id)
            filenamed = directory + str(id) + ".pkl" 
            output = load(filenamed)
            if type(output) is not dict:
                labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
                fit = dict((x,y) for x,y in zip(labels, output))
            else:
                fit = output
            if np.sum(fit['psydata']['hits']) < hit_threshold:
                raise Exception('below hit threshold')
        except:
            pass
        else:
            if cross_validation:
                lickedp = np.mean(fit['cv_pred'][fit['psydata']['y'] ==2])
                nolickp = np.mean(fit['cv_pred'][fit['psydata']['y'] ==1])
            else:
                lickedp = np.mean(fit['ypred'][fit['psydata']['y'] ==2])
                nolickp = np.mean(fit['ypred'][fit['psydata']['y'] ==1])
            ax[0].plot(nolickp,lickedp, 'o', alpha = 0.5)
            logodds.append(np.log(lickedp/nolickp))
            ids.append(id)
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
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_logodds.png")

    median = np.argsort(np.array(logodds))[len(logodds)//2]
    best = np.argmax(np.array(logodds))
    worst = np.argmin(np.array(logodds)) 
    print("Log-Odds Summary:")
    print('Worst  Session: ' + str(ids[worst]) + " " + str(logodds[worst]))
    print('Median Session: ' + str(ids[median]) + " " + str(logodds[median]))
    print('Best   Session: ' + str(ids[best]) + " " + str(logodds[best]))      

# UPDATE_REQUIRED
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

def load_fit(ID, version=None):
    '''
        Loads the fit for session ID, in directory
        Creates a dictionary for the session
        if the fit has cluster labels then it loads them and puts them into the dictionary
    '''
    directory = get_directory(version)
    filename = directory + str(ID) + ".pkl" 
    output = load(filename)
    if type(output) is not dict:
        labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
        fit = dict((x,y) for x,y in zip(labels, output))
    else:
        fit = output
    fit['ID'] = ID
    if os.path.isfile(directory+str(ID) + "_all_clusters.pkl"):
        fit['all_clusters'] = load(directory+str(ID) + "_all_clusters.pkl")
    return fit

# UPDATE_REQUIRED
def plot_cluster(ID, cluster, fit=None, directory=None):
    if directory is None:
        directory = global_directory
    if type(fit) is not dict: 
        fit = load_fit(ID, directory=directory)
    plot_fit(ID,fit=fit, cluster_labels=fit['clusters'][str(cluster)][1])

def summarize_fit(fit, version=None, savefig=False):
    directory = get_directory(version)

    fig,ax = plt.subplots(nrows=2,ncols=2, figsize=(10,7))
    my_colors = sns.color_palette("hls",len(fit['weights'].keys()))

    # Plot average weight
    means = np.mean(fit['wMode'],1)
    stds = np.std(fit['wMode'],1)
    weights_list = clean_weights(get_weights_list(fit['weights']))
    for i in np.arange(0,len(means)):
        if np.mod(i,2) == 0:
            ax[0,0].axvspan(i-.5,i+.5,color='k', alpha=0.1)
    for i in range(0,len(means)):
        ax[0,0].plot(i,means[i],'o',color=my_colors[i],label=weights_list[i])
        ax[0,0].plot([i,i],[means[i]-stds[i],means[i]+stds[i]],'-',color=my_colors[i])
    ax[0,0].set_ylabel('Average Weight')
    ax[0,0].set_xlabel('Strategy')
    ax[0,0].axhline(0,linestyle='--',color='k',alpha=0.5)
    ax[0,0].set_xlim(-0.5,len(means)-0.5)
    ax[0,0].set_xticks(np.arange(0,len(means)))

    # Plot smoothing prior
    for i in np.arange(0,len(means)):
        if np.mod(i,2) == 0:
            ax[0,1].axvspan(i-.5,i+.5,color='k', alpha=0.1)
    ax[0,1].axhline(0,linestyle='-',color='k',    alpha=0.3)
    ax[0,1].axhline(0.1,linestyle='-',color='k',  alpha=0.3)
    ax[0,1].axhline(0.01,linestyle='-',color='k', alpha=0.3)
    ax[0,1].axhline(0.001,linestyle='-',color='k',alpha=0.3)
    for i in range(0,len(means)):
        ax[0,1].plot(i,fit['hyp']['sigma'][i],'o',color=my_colors[i],label=weights_list[i])
    ax[0,1].set_ylabel('Smoothing Prior, $\sigma$ \n <-- More Smooth      More Variable -->')
    ax[0,1].set_yscale('log')
    ax[0,1].set_xlabel('Strategy')
    ax[0,1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0,1].set_xlim(-0.5,len(means)-0.5)
    ax[0,1].set_xticks(np.arange(0,len(means)))

    # plot dropout
    dropout_dict = get_session_dropout(fit)
    dropout = [dropout_dict[x] for x in sorted(list(fit['weights'].keys()))] 
    for i in np.arange(0,len(dropout)):
        if np.mod(i,2) == 0:
            ax[1,0].axvspan(i-.5,i+.5,color='k', alpha=0.1)
        ax[1,0].plot(i,dropout[i],'o',color=my_colors[i])       
    ax[1,0].axhline(0,linestyle='--',color='k',    alpha=0.3)
    ax[1,0].set_ylabel('Dropout')
    ax[1,0].set_xlabel('Model Component')
    ax[1,0].tick_params(axis='both',labelsize=10)
    ax[1,0].set_xticks(np.arange(0,len(dropout)))
    labels = sorted(list(fit['weights'].keys())) 
    ax[1,0].set_xticklabels(weights_list,rotation=90)
    
    # Plot roc
    for spine in ax[1,1].spines.values():
        spine.set_visible(False)
    ax[1,1].set_yticks([])
    ax[1,1].set_xticks([])
    roc_cv    = compute_model_roc(fit,cross_validation=True)
    roc_train = compute_model_roc(fit,cross_validation=False)
    fs= 12
    starty = 0.5
    offset = 0.04
    fig.text(.7,starty-offset*0,"Session:  "   ,fontsize=fs,horizontalalignment='right');           fig.text(.7,starty-offset*0,str(fit['ID']),fontsize=fs)
    fig.text(.7,starty-offset*1,"Driver Line:  " ,fontsize=fs,horizontalalignment='right');         fig.text(.7,starty-offset*1,fit['metadata']['driver_line'][-1],fontsize=fs)
    fig.text(.7,starty-offset*2,"Stage:  "     ,fontsize=fs,horizontalalignment='right');           fig.text(.7,starty-offset*2,str(fit['metadata']['session_type']),fontsize=fs)
    fig.text(.7,starty-offset*3,"ROC Train:  ",fontsize=fs,horizontalalignment='right');            fig.text(.7,starty-offset*3,str(round(roc_train,2)),fontsize=fs)
    fig.text(.7,starty-offset*4,"ROC CV:  "    ,fontsize=fs,horizontalalignment='right');           fig.text(.7,starty-offset*4,str(round(roc_cv,2)),fontsize=fs)
    fig.text(.7,starty-offset*5,"Lick Fraction:  ",fontsize=fs,horizontalalignment='right');        fig.text(.7,starty-offset*5,str(round(get_lick_fraction(fit),2)),fontsize=fs)
    fig.text(.7,starty-offset*6,"Lick Hit Fraction:  ",fontsize=fs,horizontalalignment='right');    fig.text(.7,starty-offset*6,str(round(get_hit_fraction(fit),2)),fontsize=fs)
    fig.text(.7,starty-offset*7,"Trial Hit Fraction:  ",fontsize=fs,horizontalalignment='right');   fig.text(.7,starty-offset*7,str(round(get_trial_hit_fraction(fit),2)),fontsize=fs)
    fig.text(.7,starty-offset*8,"Dropout Task/Timing Index:  " ,fontsize=fs,horizontalalignment='right');   fig.text(.7,starty-offset*8,str(round(get_timing_index_fit(fit),2)),fontsize=fs) 
    fig.text(.7,starty-offset*9,"Weight Task/Timing Index:  " ,fontsize=fs,horizontalalignment='right');   fig.text(.7,starty-offset*9,str(round(get_weight_timing_index_fit(fit),2)),fontsize=fs)  
    fig.text(.7,starty-offset*10,"Num Hits:  " ,fontsize=fs,horizontalalignment='right');                   fig.text(.7,starty-offset*10,np.sum(fit['psydata']['hits']),fontsize=fs)  

    plt.tight_layout()
    if savefig:
        filename = directory + 'figures_sessions/'+str(fit['ID'])+"_summary.png"
        plt.savefig(filename)
    

def plot_fit(ID, cluster_labels=None,fit=None, version=None,savefig=False,num_clusters=None):
    '''
        Plots the fit associated with a session ID
        Needs the fit dictionary. If you pass these values into, the function is much faster 
    '''

    directory = get_directory(version)
    if fit is None:
        fit = load_fit(ID, version=version)
    if savefig:
        filename = directory + str(ID)
    else:
        filename=None

    plot_weights(fit['wMode'], fit['weights'],fit['psydata'],
        errorbar=fit['credibleInt'], ypred = fit['ypred'],
        cluster_labels=cluster_labels,plot_trials=True,
        filename=filename,num_clusters=num_clusters)

    summarize_fit(fit,version=version, savefig=savefig)

    return fit
  
# UPDATE_REQUIRED 
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

# UPDATE_REQUIRED
def cluster_weights(wMode,num_clusters):
    '''
        Clusters the weights in wMode into num_clusters clusters
    '''
    output = k_means(transform(wMode.T),num_clusters)
    return output

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
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
    

# UPDATE_REQUIRED
def load_mouse(mouse, get_behavior=False):
    '''
        Takes a mouse donor_id, returns a list of all sessions objects, their IDS, and whether it was active or not. 
        if get_behavior, returns all BehaviorSessions
        no matter what, always returns the behavior_session_id for each session. 
        if global OPHYS, then forces get_behavior=False
    '''
    return pgt.load_mouse(mouse, get_behavior=get_behavior)

# UPDATE_REQUIRED
def format_mouse(sessions,IDS,format_options={}):
    '''
        Takes a list of sessions and returns a list of psydata formatted dictionaries for each session, and IDS a list of the IDS that go into each session
    '''
    d =[]
    good_ids =[]
    for session, id in zip(sessions,IDS):
        try:
            pm.annotate_licks(session) 
            pm.annotate_bouts(session)
            format_options = get_format_options(format_options)
            psydata = format_session(session,format_options)
        except Exception as e:
            print(str(id) +" "+ str(e))
        else:
            print(str(id))
            d.append(psydata)
            good_ids.append(id)
    return d, good_ids

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
def process_mouse(donor_id,directory=None,format_options={}):
    '''
        Takes a mouse donor_id, loads all ophys_sessions, and fits the model in the temporal order in which the data was created.
    '''
    if type(directory) == type(None):
        print('Couldnt find directory, using global')
        directory = global_directory

    filename = directory + 'mouse_' + str(donor_id) 
    print(filename)

    if os.path.isfile(filename+".pkl"):
        print('Already completed this fit, quitting')
        return

    print('Building List of Sessions and pulling')
    sessions, all_IDS,active = load_mouse(donor_id) # sorts the sessions by time
    print('Got  ' + str(len(all_IDS)) + ' sessions')
    print("Formating Data")
    psydatas, good_IDS = format_mouse(np.array(sessions)[active],np.array(all_IDS)[active],format_options={})
    print('Got  ' + str(len(good_IDS)) + ' good sessions')
    print("Merging Formatted Sessions")
    psydata = merge_datas(psydatas)

    print("Initial Fit")    
    strategies={BIAS,TASK0,TIMING1D,OMISSIONS, OMISSIONS1}
    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,OMISSIONS=True)
    ypred,ypred_each = compute_ypred(psydata, wMode,weights)
    plot_weights(wMode, weights,psydata,errorbar=credibleInt, ypred = ypred,filename=filename, session_labels = psydata['session_label'])

    print("Cross Validation Analysis")
    #xval_logli, xval_pL = crossValidate(psydata, hyp, weights, F=10)
    #cross_results = compute_cross_validation(psydata, hyp, weights,folds=10)
    #cv_pred = compute_cross_validation_ypred(psydata, cross_results,ypred)
    cross_results = (xval_logli, xval_pL)
    cv_pred = 0

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
    fit = cluster_mouse_fit(fit,directory=directory)

    save(filename+".pkl", fit)
    plt.close('all')

# UPDATE_REQUIRED
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
            if np.sum(summary[7]['psydata']['hits']) >= hit_threshold:
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
    data_smooth = pgt.moving_mean(data,data_mov)
    ypred_smooth = pgt.moving_mean(model,fit_mov)

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
        alarms,hits,thresholds = metrics.roc_curve(data,model)
        plt.plot(alarms,hits,'ko-')
        plt.plot([0,1],[0,1],'k--')
        plt.ylabel('Hits')
        plt.xlabel('False Alarms')
    return metrics.roc_auc_score(data,model)

def plot_session_summary_roc(IDS,version=None,savefig=False,group_label="",verbose=True,cross_validation=True,fs1=12,fs2=12,filetype=".png"):
    '''
        Make a summary plot of the histogram of AU.ROC values for all sessions in IDS.
    '''
    directory=get_directory(version)
    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    scores = []
    ids = []
    counter = 0
    hits = []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            fit = session_summary[7]
            roc = compute_model_roc(fit,plot_this=False,cross_validation=cross_validation)
            scores.append(roc)
            ids.append(id)
            hits.append(np.sum(fit['psydata']['hits']))
            counter +=1

    if counter == 0:
        print('NO DATA')
        return
    ax.set_xlim(0.5,1)
    ax.hist(np.array(scores),bins=25)
    ax.set_ylabel('Count', fontsize=fs1)
    ax.set_xlabel('ROC-AUC', fontsize=fs1)
    ax.xaxis.set_tick_params(labelsize=fs2)
    ax.yaxis.set_tick_params(labelsize=fs2)
    meanscore = np.median(np.array(scores))
    ax.plot(meanscore, ax.get_ylim()[1],'rv')
    ax.axvline(meanscore,color='r', alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"roc"+filetype)
    if verbose:
        median = np.argsort(np.array(scores))[len(scores)//2]
        best = np.argmax(np.array(scores))
        worst = np.argmin(np.array(scores)) 
        print("ROC Summary:")
        print('Worst  Session: ' + str(ids[worst]) + " " + str(scores[worst]))
        print('Median Session: ' + str(ids[median]) + " " + str(scores[median]))
        print('Best   Session: ' + str(ids[best]) + " " + str(scores[best]))     

    plt.figure()
    plt.plot(scores, hits, 'ko')
    plt.xlim(0.5,1)
    plt.ylim(0,200)
    plt.ylabel('Hits',fontsize=12)
    plt.xlabel('ROC-AUC',fontsize=12)
    plt.gca().xaxis.set_tick_params(labelsize=12)
    plt.gca().yaxis.set_tick_params(labelsize=12)    
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"roc_vs_hits"+filetype)
    return scores, ids 

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
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
    crashed = 0
    for id in ids:
        try:
            fit = load_fit(id,directory)
            w.append(fit['wMode'])
            w_ids.append(id)
        except:
            print(str(id)+" crash")
            crashed+=1
            pass
    print(str(crashed) +" crashed sessions")
    return w, w_ids

# UPDATE_REQUIRED
def merge_weights(w): 
    '''
        Merges a list of weights into one long array of weights
    '''
    return np.concatenate(w,axis=1)           

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
def save_session_clusters(session_clusters, directory=None):
    '''
        Saves the session_clusters in 'session_clusters,pkl'

    '''
    if type(directory) == type(None):
        directory = global_directory

    filename = directory + "session_clusters.pkl"
    save(filename,session_clusters)

# UPDATE_REQUIRED
def save_all_clusters(w_ids,session_clusters, directory=None):
    '''
        Saves each sessions all_clusters
    '''
    if type(directory) == type(None):
        directory = global_directory

    for key in session_clusters.keys():
        filename = directory + str(key) + "_all_clusters.pkl" 
        save(filename, session_clusters[key]) 

# UPDATE_REQUIRED
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

def check_session(ID, version=None):
    '''
        Checks if the ID has a model fit computed
    '''
    directory=get_directory(version)

    filename = directory + str(ID) + ".pkl" 
    has_fit =  os.path.isfile(filename)

    if has_fit:
        print("Session has a fit, load the results with load_fit(ID)")
    else:
        print("Session does not have a fit, fit the session with process_session(ID)")

    return has_fit

def get_all_dropout(IDS,version=None,hit_threshold=0,verbose=False): 
    '''
        For each session in IDS, returns the vector of dropout scores for each model
    '''

    directory=get_directory(version)

    all_dropouts = []
    hits = []
    false_alarms = []
    correct_reject = []
    misses = []
    ids = []
    crashed = 0
    low_hits = 0
    
    # Loop through IDS, add information from sessions above hit threshold
    for id in tqdm(IDS):
        try:
            fit = load_fit(id,version=version)
            if np.sum(fit['psydata']['hits']) >= hit_threshold:
                dropout_dict = get_session_dropout(fit)
                dropout = [dropout_dict[x] for x in sorted(list(fit['weights'].keys()))] 
                all_dropouts.append(dropout)
                hits.append(np.sum(fit['psydata']['hits']))
                false_alarms.append(np.sum(fit['psydata']['false_alarms']))
                correct_reject.append(np.sum(fit['psydata']['correct_reject']))
                misses.append(np.sum(fit['psydata']['misses']))
                ids.append(id)
            else:
                low_hits+=1
        except:
            if verbose:
                print(str(id) +" crash")
            crashed +=1

    print(str(crashed) + " crashed")
    print(str(low_hits) + " below hit threshold")
    dropouts = np.stack(all_dropouts,axis=1)
    filepath = directory + "all_dropouts.pkl"
    save(filepath, dropouts)
    return dropouts,hits, false_alarms, misses,ids, correct_reject

def load_all_dropout(version=None):
    directory = get_directory(version)
    dropout = load(directory+"all_dropouts.pkl")
    return dropout

# UPDATE_REQUIRED
def get_mice_weights(mice_ids,version=None,hit_threshold=0,verbose=False,manifest = None):
    directory=get_directory(version)
    if manifest is None:
        manifest = pgt.get_ophys_manifest()
    mice_weights = []
    mice_good_ids = []
    crashed = 0
    low_hits = 0
    # Loop through IDS
    for id in tqdm(mice_ids):
        this_mouse = []
        for sess in manifest.query('donor_id == @id').query('active').behavior_session_id.values:
            try:
                fit = load_fit(sess,version=version)
                if np.sum(fit['psydata']['hits']) >= hit_threshold:
                    this_mouse.append(np.mean(fit['wMode'],1))
                else:
                    low_hits +=1
            except:
                if verbose:
                    print("Mouse: "+str(id)+" session: "+str(sess) +" crash")
                crashed += 1
        if len(this_mouse) > 0:
            this_mouse = np.stack(this_mouse,axis=1)
            mice_weights.append(this_mouse)
            mice_good_ids.append(id)
    print()
    print(str(crashed) + " crashed")
    print(str(low_hits) + " below hit_threshold")
    return mice_weights,mice_good_ids

def get_mice_dropout(mice_ids,version=None,hit_threshold=0,verbose=False,manifest=None):

    directory=get_directory(version)    
    if manifest is None:
        manifest = pgt.get_ophys_manifest()

    mice_dropouts = []
    mice_good_ids = []
    crashed = 0
    low_hits = 0

    # Loop through IDS
    for id in tqdm(mice_ids):
        this_mouse = []
        for sess in manifest.query('donor_id ==@id').query('active')['behavior_session_id'].values:
            try:
                fit = load_fit(sess,version=version)
                if np.sum(fit['psydata']['hits']) >= hit_threshold:
                    dropout_dict = get_session_dropout(fit)
                    dropout = [dropout_dict[x] for x in sorted(list(fit['weights'].keys()))] 
                    this_mouse.append(dropout)
                else:
                    low_hits +=1
            except:
                if verbose:
                    print("Mouse: "+str(id)+" Session:"+str(sess)+" crash")
                crashed +=1
        if len(this_mouse) > 0:
            this_mouse = np.stack(this_mouse,axis=1)
            mice_dropouts.append(this_mouse)
            mice_good_ids.append(id)
    print()
    print(str(crashed) + " crashed")
    print(str(low_hits) + " below hit_threshold")

    return mice_dropouts,mice_good_ids

def PCA_dropout(ids,mice_ids,version,verbose=False,hit_threshold=0,manifest=None):
    dropouts, hits,false_alarms,misses,ids,correct_reject = get_all_dropout(ids,
        version,verbose=verbose,hit_threshold=hit_threshold)

    mice_dropouts, mice_good_ids = get_mice_dropout(mice_ids,
        version=version,verbose=verbose,hit_threshold=hit_threshold,
        manifest = manifest)

    fit = load_fit(ids[1],version=version)
    labels = sorted(list(fit['weights'].keys()))
    pca,dropout_dex,varexpl = PCA_on_dropout(dropouts, labels=labels,
        mice_dropouts=mice_dropouts,mice_ids=mice_good_ids, hits=hits,
        false_alarms=false_alarms, misses=misses,version=version, correct_reject = correct_reject)

    return dropout_dex,varexpl

def PCA_on_dropout(dropouts,labels=None,mice_dropouts=None, mice_ids = None,hits=None,false_alarms=None, misses=None,version=None,fs1=12,fs2=12,filetype='.png',ms=2,correct_reject=None):
    directory=get_directory(version)
    if directory[-3:-1] == '12':
        sdex = 2
        edex = 6
    elif directory[-2] == '2':
        sdex = 2
        edex = 16
    elif directory[-2] == '4':
        sdex = 2
        edex = 18
    elif directory[-2] == '6':
        sdex = 2 
        edex = 6
    elif directory[-2] == '7':
        sdex = 2 
        edex = 6
    elif directory[-2] == '8':
        sdex = 2 
        edex = 6
    elif directory[-2] == '9':
        sdex = 2 
        edex = 6
    elif directory[-3:-1] == '10':
        sdex = 2
        edex = 6
    elif version == 20: 
        sdex = np.where(np.array(labels) == 'task0')[0][0]
        edex = np.where(np.array(labels) == 'timing1D')[0][0]
    dex = -(dropouts[sdex,:] - dropouts[edex,:])

    
    # Removing Bias from PCA
    dropouts = dropouts[1:,:]
    labels = labels[1:]

    # Do pca
    pca = PCA()
    pca.fit(dropouts.T)
    X = pca.transform(dropouts.T)
    
    fig,ax = plt.subplots(figsize=(6,4.5)) # FIG1
    fig=plt.gcf()
    ax = [plt.gca()]
    scat = ax[0].scatter(-X[:,0], X[:,1],c=dex,cmap='plasma')
    cbar = fig.colorbar(scat, ax = ax[0])
    cbar.ax.set_ylabel('Strategy Dropout Index',fontsize=fs2)
    ax[0].set_xlabel('Dropout PC 1',fontsize=fs1)
    ax[0].set_ylabel('Dropout PC 2',fontsize=fs1)
    ax[0].axis('equal')
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.tight_layout()   
    plt.savefig(directory+"figures_summary/dropout_pca"+filetype)
 
    plt.figure(figsize=(6,3))# FIG2
    fig=plt.gcf()
    ax.append(plt.gca())
    ax[1].axhline(0,color='k',alpha=0.2)
    for i in np.arange(0,len(dropouts)):
        if np.mod(i,2) == 0:
            ax[1].axvspan(i-.5,i+.5,color='k', alpha=0.1)
    pca1varexp = str(100*round(pca.explained_variance_ratio_[0],2))
    pca2varexp = str(100*round(pca.explained_variance_ratio_[1],2))
    ax[1].plot(-pca.components_[0,:],'ko-',label='PC1 '+pca1varexp+"%")
    ax[1].plot(-pca.components_[1,:],'ro-',label='PC2 '+pca2varexp+"%")
    ax[1].set_xlabel('Model Component',fontsize=12)
    ax[1].set_ylabel('% change in \n evidence',fontsize=12)
    ax[1].tick_params(axis='both',labelsize=10)
    ax[1].set_xticks(np.arange(0,len(dropouts)))
    if type(labels) is not type(None):    
        ax[1].set_xticklabels(labels,rotation=90)
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/dropout_pca_1.png")

    plt.figure(figsize=(5,4.5))# FIG3
    scat = plt.gca().scatter(-X[:,0],dex,c=dex,cmap='plasma')
    #cbar = plt.gcf().colorbar(scat, ax = plt.gca())
    #cbar.ax.set_ylabel('Task Dropout Index',fontsize=fs1)
    plt.gca().set_xlabel('Dropout PC 1',fontsize=fs1)
    plt.gca().set_ylabel('Strategy Dropout Index',fontsize=fs1)   
    plt.gca().axis('equal')
    plt.gca().tick_params(axis='both',labelsize=10)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/dropout_pca_3"+filetype)

    plt.figure(figsize=(5,4.5))# FIG4 
    ax = plt.gca()
    if type(mice_dropouts) is not type(None):
        ax.axhline(0,color='k',alpha=0.2)
        ax.set_xlabel('Individual Mice', fontsize=fs1)
        ax.set_ylabel('Strategy Dropout Index', fontsize=fs1)
        ax.set_xticks(range(0,len(mice_dropouts)))
        ax.set_ylim(-45,40)
        mean_drop = []
        for i in range(0, len(mice_dropouts)):
            mean_drop.append(-1*np.nanmean(mice_dropouts[i][sdex,:]-mice_dropouts[i][edex,:]))
        sortdex = np.argsort(np.array(mean_drop))
        mice_dropouts = [mice_dropouts[i] for i in sortdex]
        mean_drop = np.array(mean_drop)[sortdex]
        for i in range(0,len(mice_dropouts)):
            if np.mod(i,2) == 0:
                ax.axvspan(i-.5,i+.5,color='k', alpha=0.1)
            mouse_dex = -(mice_dropouts[i][sdex,:]-mice_dropouts[i][edex,:])
            ax.plot([i-0.5, i+0.5], [mean_drop[i],mean_drop[i]], 'k-',alpha=0.3)
            ax.scatter(i*np.ones(np.shape(mouse_dex)), mouse_dex,ms,c=mouse_dex,cmap='plasma',vmin=(dex).min(),vmax=(dex).max(),alpha=1)
        sorted_mice_ids = ["" for i in sortdex]
        ax.set_xticklabels(sorted_mice_ids,fontdict={'fontsize':10},rotation=90)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.xlim(-1,len(mice_dropouts))
    plt.savefig(directory+"figures_summary/dropout_pca_mice"+filetype)

    plt.figure(figsize=(5,4.5))
    ax = plt.gca()   
    ax.plot(pca.explained_variance_ratio_*100,'ko-')
    ax.set_xlabel('PC Dimension',fontsize=fs1)
    ax.set_ylabel('Explained Variance %',fontsize=fs1)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_var_expl"+filetype)

    fig, ax = plt.subplots(2,3,figsize=(10,6))
    #ax[0,0].axhline(0,color='k',alpha=0.2)
    #ax[0,0].axvline(0,color='k',alpha=0.2)
    ax[0,0].scatter(-X[:,0], dex,c=dex,cmap='plasma')
    ax[0,0].set_xlabel('Dropout PC 1',fontsize=fs2)
    ax[0,0].set_ylabel('Strategy Dropout Index',fontsize=fs2)
    ax[0,1].plot(pca.explained_variance_ratio_*100,'ko-')
    ax[0,1].set_xlabel('PC Dimension',fontsize=fs2)
    ax[0,1].set_ylabel('Explained Variance %',fontsize=fs2)

    if type(mice_dropouts) is not type(None):
        ax[1,0].axhline(0,color='k',alpha=0.2)
        ax[1,0].set_ylabel('Strategy Dropout Index', fontsize=12)
        ax[1,0].set_xticks(range(0,len(mice_dropouts)))
        ax[1,0].set_ylim(-45,40)
        mean_drop = []
        for i in range(0, len(mice_dropouts)):
            mean_drop.append(-1*np.nanmean(mice_dropouts[i][sdex,:]-mice_dropouts[i][edex,:]))
        sortdex = np.argsort(np.array(mean_drop))
        mice_dropouts = [mice_dropouts[i] for i in sortdex]
        mean_drop = np.array(mean_drop)[sortdex]
        for i in range(0,len(mice_dropouts)):
            if np.mod(i,2) == 0:
                ax[1,0].axvspan(i-.5,i+.5,color='k', alpha=0.1)
            mouse_dex = -(mice_dropouts[i][sdex,:]-mice_dropouts[i][edex,:])
            ax[1,0].plot([i-0.5, i+0.5], [mean_drop[i],mean_drop[i]], 'k-',alpha=0.3)
            ax[1,0].scatter(i*np.ones(np.shape(mouse_dex)), mouse_dex,c=mouse_dex,cmap='plasma',vmin=(dex).min(),vmax=(dex).max(),alpha=1)
        sorted_mice_ids = [mice_ids[i] for i in sortdex]
        ax[1,0].set_xticklabels(sorted_mice_ids,fontdict={'fontsize':10},rotation=90)
    if type(hits) is not type(None):
        ax[1,1].scatter(dex, hits,c=dex,cmap='plasma')
        ax[1,1].set_ylabel('Hits/session',fontsize=12)
        ax[1,1].set_xlabel('Strategy Dropout Index',fontsize=12)
        ax[1,1].axvline(0,color='k',alpha=0.2)
        ax[1,1].set_xlim(-45,40)
        ax[1,1].set_ylim(bottom=0)

        ax[0,2].scatter(dex, false_alarms,c=dex,cmap='plasma')
        ax[0,2].set_ylabel('FA/session',fontsize=12)
        ax[0,2].set_xlabel('Strategy Dropout Index',fontsize=12)
        ax[0,2].axvline(0,color='k',alpha=0.2)
        ax[0,2].set_xlim(-45,40)
        ax[0,2].set_ylim(bottom=0)


        ax[1,2].scatter(dex, misses,c=dex,cmap='plasma')
        ax[1,2].set_ylabel('Miss/session',fontsize=12)
        ax[1,2].set_xlabel('Strategy Dropout Index',fontsize=12)
        ax[1,2].axvline(0,color='k',alpha=0.2)
        ax[1,2].set_xlim(-45,40)
        ax[1,2].set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/dropout_pca_2.png")

    plt.figure(figsize=(5,4.5))
    ax = plt.gca() 
    ax.scatter(dex, hits,c=dex,cmap='plasma')
    ax.set_ylabel('Hits/session',fontsize=fs1)
    ax.set_xlabel('Strategy Dropout Index',fontsize=fs1)
    ax.axvline(0,color='k',alpha=0.2)
    ax.set_xlim(-45,40)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_hits"+filetype)


    plt.figure(figsize=(5,4.5))
    ax = plt.gca()
    ax.scatter(dex, false_alarms,c=dex,cmap='plasma')
    ax.set_ylabel('FA/session',fontsize=fs1)
    ax.set_xlabel('Strategy Dropout Index',fontsize=fs1)
    ax.axvline(0,color='k',alpha=0.2)
    ax.set_xlim(-45,40)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_fa"+filetype)



    plt.figure(figsize=(5,4.5))
    ax = plt.gca() 
    ax.scatter(dex, misses,c=dex,cmap='plasma')
    ax.set_ylabel('Miss/session',fontsize=fs1)
    ax.set_xlabel('Strategy Dropout Index',fontsize=fs1)
    ax.axvline(0,color='k',alpha=0.2)
    ax.set_xlim(-45,40)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_miss"+filetype)

    plt.figure(figsize=(5,4.5))
    ax = plt.gca() 
    ax.scatter(dex, correct_reject,c=dex,cmap='plasma')
    ax.set_ylabel('CR/session',fontsize=fs1)
    ax.set_xlabel('Strategy Dropout Index',fontsize=fs1)
    ax.axvline(0,color='k',alpha=0.2)
    ax.set_xlim(-45,40)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_cr"+filetype)

    varexpl = 100*round(pca.explained_variance_ratio_[0],2)
    return pca,dex,varexpl

def PCA_weights(ids,mice_ids,version=None,verbose=False,manifest = None,hit_threshold=0):
    directory=get_directory(version)
    all_weights,good_ids =plot_session_summary_weights(ids,return_weights=True,version=version,hit_threshold=hit_threshold)
    x = np.vstack(all_weights)

    fit = load_fit(ids[np.where(good_ids)[0][0]],version=version)
    weight_names = sorted(list(fit['weights'].keys()))
    task_index = np.where(np.array(weight_names) == 'task0')[0][0]
    timing_index = np.where(np.array(weight_names) == 'timing1D')[0][0]
    task = x[:,np.where(np.array(weight_names) == 'task0')[0][0]]
    timing = x[:,np.where(np.array(weight_names) == 'timing1D')[0][0]]

    dex = task-timing
    pca = PCA()
    pca.fit(x)
    X = pca.transform(x)
    plt.figure(figsize=(4,2.9))
    scat = plt.gca().scatter(X[:,0],X[:,1],c=dex,cmap='plasma')
    cbar = plt.gcf().colorbar(scat, ax = plt.gca())
    cbar.ax.set_ylabel('Strategy Weight Index',fontsize=12)
    plt.gca().set_xlabel('Weight PC 1 - '+str(100*round(pca.explained_variance_ratio_[0],2))+"%",fontsize=12)
    plt.gca().set_ylabel('Weight PC 2 - '+str(100*round(pca.explained_variance_ratio_[1],2))+"%",fontsize=12)
    plt.gca().axis('equal')   
    plt.gca().tick_params(axis='both',labelsize=10)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/weight_pca_1.png")

    plt.figure(figsize=(4,2.9))
    scat = plt.gca().scatter(X[:,0],dex,c=dex,cmap='plasma')
    cbar = plt.gcf().colorbar(scat, ax = plt.gca())
    cbar.ax.set_ylabel('Strategy Weight Index',fontsize=12)
    plt.gca().set_xlabel('Weight PC 1',fontsize=12)
    plt.gca().set_ylabel('Strategy Weight Index',fontsize=12)
    plt.gca().axis('equal')
    plt.gca().tick_params(axis='both',labelsize=10)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/weight_pca_2.png")   

    plt.figure(figsize=(6,3))
    fig=plt.gcf()
    ax =plt.gca()
    ax.axhline(0,color='k',alpha=0.2)
    for i in np.arange(0,np.shape(x)[1]):
        if np.mod(i,2) == 0:
            ax.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    pca1varexp = str(100*round(pca.explained_variance_ratio_[0],2))
    pca2varexp = str(100*round(pca.explained_variance_ratio_[1],2))
    ax.plot(pca.components_[0,:],'ko-',label='PC1 '+pca1varexp+"%")
    ax.plot(pca.components_[1,:],'ro-',label='PC2 '+pca2varexp+"%")
    ax.set_xlabel('Model Component',fontsize=12)
    ax.set_ylabel('Avg Weight',fontsize=12)
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xticks(np.arange(0,np.shape(x)[1]))
    weights_list = get_weights_list(fit['weights'])
    labels = clean_weights(weights_list)    
    ax.set_xticklabels(labels,rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/weight_pca_3.png")

    _, hits,false_alarms,misses,ids = get_all_dropout(ids,version=version,verbose=verbose,hit_threshold=hit_threshold)
    mice_weights, mice_good_ids = get_mice_weights(mice_ids, version=version,verbose=verbose,manifest = manifest)

    fig, ax = plt.subplots(2,3,figsize=(10,6))
    ax[0,0].scatter(X[:,0], dex,c=dex,cmap='plasma')
    ax[0,0].set_xlabel('Weight PC 1',fontsize=12)
    ax[0,0].set_ylabel('Strategy Weight Index',fontsize=12)
    ax[0,1].plot(pca.explained_variance_ratio_*100,'ko-')
    ax[0,1].set_xlabel('PC Dimension',fontsize=12)
    ax[0,1].set_ylabel('Explained Variance %',fontsize=12)

    ax[1,0].axhline(0,color='k',alpha=0.2)
    ax[1,0].set_ylabel('Strategy Weight Index', fontsize=12)
    ax[1,0].set_xticks(range(0,len(mice_good_ids)))
    ax[1,0].set_ylim(-8,8)
    mean_weight = []
    for i in range(0, len(mice_good_ids)):
        this_weight = np.mean(mice_weights[i],1)
        mean_weight.append(this_weight[task_index] -this_weight[timing_index])
    sortdex = np.argsort(np.array(mean_weight))
    mice_weights_sorted = [mice_weights[i] for i in sortdex]
    mean_weight = np.array(mean_weight)[sortdex]
    for i in range(0,len(mice_good_ids)):
        if np.mod(i,2) == 0:
            ax[1,0].axvspan(i-.5,i+.5,color='k', alpha=0.1)
        this_mouse_weights = mice_weights_sorted[i][task_index,:] - mice_weights_sorted[i][timing_index,:]
        ax[1,0].plot([i-0.5,i+0.5],[mean_weight[i],mean_weight[i]],'k-',alpha=0.3)
        ax[1,0].scatter(i*np.ones(np.shape(this_mouse_weights)), this_mouse_weights,c=this_mouse_weights,cmap='plasma',vmin=(dex).min(),vmax=(dex).max(),alpha=1)
    sorted_mice_ids = [mice_good_ids[i] for i in sortdex]
    ax[1,0].set_xticklabels(sorted_mice_ids,fontdict={'fontsize':10},rotation=90) 
    ax[1,1].scatter(dex, hits,c=dex,cmap='plasma')
    ax[1,1].set_ylabel('Hits/session',fontsize=12)
    ax[1,1].set_xlabel('Strategy Weight Index',fontsize=12)
    ax[1,1].axvline(0,color='k',alpha=0.2)
    ax[1,1].set_xlim(-8,8)
    ax[1,1].set_ylim(bottom=0)

    ax[0,2].scatter(dex, false_alarms,c=dex,cmap='plasma')
    ax[0,2].set_ylabel('FA/session',fontsize=12)
    ax[0,2].set_xlabel('Strategy Weight Index',fontsize=12)
    ax[0,2].axvline(0,color='k',alpha=0.2)
    ax[0,2].set_xlim(-8,8)
    ax[0,2].set_ylim(bottom=0)

    ax[1,2].scatter(dex, misses,c=dex,cmap='plasma')
    ax[1,2].set_ylabel('Miss/session',fontsize=12)
    ax[1,2].set_xlabel('Strategy Weight Index',fontsize=12)
    ax[1,2].axvline(0,color='k',alpha=0.2)
    ax[1,2].set_xlim(-8,8)
    ax[1,2].set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/weight_pca_4.png")

    varexpl =100*round(pca.explained_variance_ratio_[0],2)
    return dex, varexpl


def PCA_analysis(ids, mice_ids,version,hit_threshold=0,manifest=None):
    # PCA on dropouts
    drop_dex,drop_varexpl = PCA_dropout(ids,mice_ids,version,hit_threshold=hit_threshold,manifest=manifest)

    # PCA on weights
    weight_dex,weight_varexpl = PCA_weights(ids,mice_ids,version,manifest=manifest)
   
    # Compare
    directory=get_directory(version) 
    plt.figure(figsize=(5,4.5))
    scat = plt.gca().scatter(weight_dex,drop_dex,c=weight_dex, cmap='plasma')
    plt.gca().set_xlabel('Task Weight Index' ,fontsize=24)
    plt.gca().set_ylabel('Task Dropout Index',fontsize=24)
    #cbar = plt.gcf().colorbar(scat, ax = plt.gca())
    #cbar.ax.set_ylabel('Task Weight Index',fontsize=20)
    plt.gca().tick_params(axis='both',labelsize=10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/dropout_vs_weight_pca_1.svg")

# UPDATE_REQUIRED
def compare_versions(directories, IDS):
    all_rocs = []
    for d in directories:
        my_rocs = []
        for id in tqdm(IDS):
            try:
                fit = load_fit(id, directory=d)
                my_rocs.append(compute_model_roc(fit,cross_validation=True))
            except:
                pass
        all_rocs.append(my_rocs)
    return all_rocs

# UPDATE_REQUIRED
def compare_versions_plot(all_rocs):
    plt.figure()
    plt.ylabel('ROC')
    plt.xlabel('Model Version')
    plt.ylim(0.75,.85)
    for index, roc in enumerate(all_rocs):
        plt.plot(index, np.mean(roc),'ko')

# UPDATE_REQUIRED
def compare_fits(ID, directories,cv=True):
    fits = []
    roc = []
    for d in directories:
        print(d)
        fits.append(load_fit(ID,directory=d))
        roc.append(compute_model_roc(fits[-1],cross_validation=cv))
    return fits,roc
    
# UPDATE_REQUIRED
def compare_all_fits(IDS, directories,cv=True):
    all_fits = []
    all_roc = []
    all_ids = []
    for id in IDS:
        print(id)
        try:
            fits, roc = compare_fits(id,directories,cv=cv)
            all_fits.append(fits)
            all_roc.append(roc)
            all_ids.append(id)
        except:
            print(" crash")
    filename = directories[1] + "all_roc.pkl"
    save(filename,[all_ids,all_roc])
    return all_roc

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
def compare_roc_session_mouse(fit,directory):
    # Asking how different the ROC fits are with mouse fits
    fit['roc_session_individual'] = []
    for id in fit['good_IDS']:
        print(id)
        try:
            sfit = load_fit(id[6:],directory=directory)
            data = copy.copy(sfit['psydata']['y']-1)
            model =copy.copy(sfit['cv_pred'])
            fit['roc_session_individual'].append(metrics.roc_auc_score(data,model))
        except:
            fit['roc_session_individual'].append(np.nan)
        
# UPDATE_REQUIRED
def mouse_roc(fit):
    fit['roc_session'] = []
    for i in range(0,len(fit['psydata']['dayLength'])):
        data = copy.copy(fit['psydata_session'][i]-1)
        model = copy.copy(fit['cv_pred_session'][i])
        fit['roc_session'].append(metrics.roc_auc_score(data,model))

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
def plot_all_mouse_session_roc(directory):
    rocs = load(directory+"all_roc_session_mouse.pkl")
    plt.figure()
    plt.plot(np.array(rocs[1])*100, np.array(rocs[0])*100,'ko')
    plt.plot([60,100],[60,100],'k--')
    plt.xlabel('Session ROC (%)')
    plt.ylabel('Mouse ROC (%)')
    plt.savefig(directory+"all_roc_session_mouse.png") 

# UPDATE_REQUIRED
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

# UPDATE_REQUIRED
def plot_mouse_roc_comparisons(directory,label1="", label2=""):
    rocs = load(directory + "all_roc_mouse_comparison.pkl")
    plt.figure(figsize=(5.75,5))
    plt.plot(np.array(rocs[1])*100, np.array(rocs[0])*100,'ko')
    plt.plot([50,100],[50,100],'k--')
    plt.xlabel(label2+' ROC (%)')
    plt.ylabel(label1+' ROC (%)')
    plt.ylim([50,100])
    plt.xlim([50,100])
    plt.savefig(directory+"figures_summary/all_roc_mouse_comparison.png")

# UPDATE_REQUIRED
def get_session_task_index(id):
    raise Exception('outdated')
    fit = load_fit(id)
    #dropout = np.empty((len(fit['models']),))
    #for i in range(0,len(fit['models'])):
    #    dropout[i] = (1-fit['models'][i][1]/fit['models'][0][1])*100
    dropout = get_session_dropout(fit)
    model_dex = -(dropout[2] - dropout[16]) ### BUG?
    return model_dex

# UPDATE_REQUIRED
def hazard_index(IDS,directory,sdex = 2, edex = 6):
    dexes =[]
    for count, id in enumerate(tqdm(IDS)):
        try:
            fit = load_fit(id,directory=directory)
            #dropout = np.empty((len(fit['models']),))
            #for i in range(0,len(fit['models'])):
            #    dropout[i] = (1-fit['models'][i][1]/fit['models'][0][1])*100
            dropout = get_session_dropout(fit)
            model_dex = -(dropout[2] - dropout[6])
            session = pgt.get_data(id)
            pm.annotate_licks(session)
            bout = pt.get_bout_table(session) 
            hazard_hits, hazard_miss = pt.get_hazard(bout, None, nbins=15) 
            hazard_dex = np.sum(hazard_miss - hazard_hits)
            
            dexes.append([model_dex, hazard_dex])
        except:
            print(' crash')
    return dexes

# UPDATE_REQUIRED
def plot_hazard_index(dexes):
    plt.figure(figsize=(5,4))
    ax = plt.gca()
    dex = np.vstack(dexes)
    ax.scatter(dex[:,0],dex[:,1],c=-dex[:,0],cmap='plasma')
    ax.axvline(0,color='k',alpha=0.2)
    ax.axhline(0,color='k',alpha=0.2)
    ax.set_xlabel('Model Index (Task-Timing) \n <-- more timing      more task -->',fontsize=12)
    ax.set_ylabel('Hazard Function Index',fontsize=12)
    ax.set_xlim([-20, 20])
    plt.tight_layout()

def get_weight_timing_index_fit(fit):
    '''
        Return Task/Timing Index from average weights
    '''
    weights = get_weights_list(fit['weights'])
    wMode = fit['wMode']
    avg_weight_task   = np.mean(wMode[np.where(np.array(weights) == 'task0')[0][0],:])
    avg_weight_timing = np.mean(wMode[np.where(np.array(weights) == 'timing1D')[0][0],:])
    index = avg_weight_task - avg_weight_timing
    return index
    

def get_timing_index(id, version,return_all=False):

    try:
        fit = load_fit(id,version=version)
        return get_timing_index_fit(fit,return_all=return_all)
    except:
        if return_all:
            return np.nan, np.nan, np.nan
        else:
            return np.nan

def get_timing_index_fit(fit,return_all=False):
    dropout = get_session_dropout(fit)
    model_dex = -(dropout['task0'] - dropout['timing1D'])
    if return_all:
        return model_dex, dropout['task0'], dropout['timing1D']
    else:
        return model_dex   
 
def get_session_dropout(fit, cross_validation=False):
    dropout = dict()
    models = sorted(list(fit['models'].keys()))
    models.remove('Full')
    if cross_validation:
         for m in models:
            dropout[m] = (1-get_cross_validation_dropout(fit['models'][m][6])/get_cross_validation_dropout(fit['models']['Full'][6]))*100   
    else:
        for m in models:
            dropout[m] = (1-fit['models'][m][1]/fit['models']['Full'][1])*100
    
    return dropout   

def get_lick_fraction(fit,first_half=False, second_half=False):
    if first_half:
        numflash = len(fit['psydata']['y'][fit['psydata']['flash_ids'] < 2400])
        numbouts = np.sum(fit['psydata']['y'][fit['psydata']['flash_ids'] < 2400] -1)
        if numflash == 0:
            numflash = 1
        return numbouts/numflash    
    elif second_half:
        numflash = len(fit['psydata']['y'][fit['psydata']['flash_ids'] >= 2400])
        numbouts = np.sum(fit['psydata']['y'][fit['psydata']['flash_ids'] >= 2400]-1)
        if numflash == 0:
            numflash = 1
        return numbouts/numflash 
    else:
        numflash = len(fit['psydata']['y'])
        numbouts = np.sum(fit['psydata']['y']-1)
        if numflash == 0:
            numflash = 1
        return numbouts/numflash 
 
def get_hit_fraction(fit,first_half=False, second_half=False):
    if first_half:
        numhits = np.sum(fit['psydata']['hits'][fit['psydata']['flash_ids'] < 2400])
        numbouts = np.sum(fit['psydata']['y'][fit['psydata']['flash_ids'] < 2400]-1)
        if numbouts ==0:
            numbouts = 1
        return numhits/numbouts       
    elif second_half:
        numhits = np.sum(fit['psydata']['hits'][fit['psydata']['flash_ids'] >= 2400])
        numbouts = np.sum(fit['psydata']['y'][fit['psydata']['flash_ids'] >= 2400]-1)
        if numbouts ==0:
            numbouts = 1
        return numhits/numbouts    
    else:
        numhits = np.sum(fit['psydata']['hits'])
        numbouts = np.sum(fit['psydata']['y']-1)
        if numbouts ==0:
            numbouts = 1
        return numhits/numbouts    

def get_trial_hit_fraction(fit,first_half=False, second_half=False):
    if first_half:
        numhits = np.sum(fit['psydata']['hits'][fit['psydata']['flash_ids'] < 2400])
        nummiss = np.sum(fit['psydata']['misses'][fit['psydata']['flash_ids'] < 2400])
        if numhits+nummiss == 0:
            nummiss = 1
        return numhits/(numhits+nummiss)   
    elif second_half:
        numhits = np.sum(fit['psydata']['hits'][fit['psydata']['flash_ids'] >= 2400])
        nummiss = np.sum(fit['psydata']['misses'][fit['psydata']['flash_ids'] >= 2400])
        if numhits+nummiss == 0:
            nummiss = 1
        return numhits/(numhits+nummiss)
    else:
        numhits = np.sum(fit['psydata']['hits'])
        nummiss = np.sum(fit['psydata']['misses'])
        if numhits+nummiss == 0:
            nummiss = 1
        return numhits/(numhits+nummiss)

def get_all_timing_index(ids, version,hit_threshold=0):
    directory=get_directory(version)
    df = pd.DataFrame(data={'Task/Timing Index':[],'taskdex':[],'timingdex':[],'numlicks':[],'behavior_session_id':[]})
    crashed = 0
    low_hits = 0
    for id in ids:
        try:
            fit = load_fit(id, version=version)
            if np.sum(fit['psydata']['hits']) >= hit_threshold:
                model_dex, taskdex,timingdex = get_timing_index_fit(fit,return_all=True)
                numlicks = np.sum(fit['psydata']['y']-1) 
                d = {'Strategy Index':model_dex,'taskdex':taskdex,'timingdex':timingdex,'numlicks':numlicks,'behavior_session_id':id}
                df = df.append(d,ignore_index=True)
            else:
                low_hits +=1
        except:
            crashed+=1
    print(str(crashed) + " crashed")
    print(str(low_hits) + " below hit_threshold")
    return df.set_index('behavior_session_id')

def plot_model_index_summaries(df,version):

    directory=get_directory(version)
    fig, ax = plt.subplots(figsize=(6,4.5))
    scat = ax.scatter(-df.taskdex, -df.timingdex,c=df['Strategy Index'],cmap='plasma')
    ax.set_ylabel('Timing Dropout',fontsize=24)
    ax.set_xlabel('Visual Dropout',fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    cbar = fig.colorbar(scat, ax = ax)
    cbar.ax.set_ylabel('Strategy Dropout Index',fontsize=20)
    plt.tight_layout()
    plt.savefig(directory+'figures_summary/timing_vs_task_breakdown_1.svg')
    plt.savefig(directory+'figures_summary/timing_vs_task_breakdown_1.png')

    fig, ax = plt.subplots(nrows=2,ncols=2,figsize=(8,5))
    scat = ax[0,0].scatter(-df.taskdex, -df.timingdex,c=df['Strategy Index'],cmap='plasma')
    ax[0,0].set_ylabel('Timing Dex')
    ax[0,0].set_xlabel('Visual Dex')
    cbar = fig.colorbar(scat, ax = ax[0,0])
    cbar.ax.set_ylabel('Strategy Index',fontsize=12)

    scat = ax[0,1].scatter(df['Strategy Index'], df['numlicks'],c=df['Strategy Index'],cmap='plasma')
    ax[0,1].set_xlabel('Strategy Index')
    ax[0,1].set_ylabel('Number Lick Bouts')
    cbar = fig.colorbar(scat, ax = ax[0,1])
    cbar.ax.set_ylabel('Strategy Index',fontsize=12)
    
    scat = ax[1,0].scatter(-df['taskdex'],df['numlicks'],c=df['Strategy Index'],cmap='plasma')
    ax[1,0].set_xlabel('Visual Dex')
    ax[1,0].set_ylabel('Number Lick Bouts')
    cbar = fig.colorbar(scat, ax = ax[1,0])
    cbar.ax.set_ylabel('Strategy Index',fontsize=12)

    scat = ax[1,1].scatter(-df['timingdex'],df['numlicks'],c=df['Strategy Index'],cmap='plasma')
    ax[1,1].set_xlabel('Timing Dex')
    ax[1,1].set_ylabel('Number Lick Bouts')
    cbar = fig.colorbar(scat, ax = ax[1,1])
    cbar.ax.set_ylabel('Strategy Index',fontsize=12)
    plt.tight_layout()
    plt.savefig(directory+'figures_summary/timing_vs_task_breakdown.png')
    plt.savefig(directory+'figures_summary/timing_vs_task_breakdown.svg')

def compute_model_roc_timing(fit,plot_this=False):
    '''
        Computes area under the ROC curve for the model in fit. If plot_this, then plots the ROC curve. 
        If cross_validation, then uses the cross validated prediction in fit, not he training fit.
        Returns the AU. ROC single float
    '''

    data = copy.copy(fit['psydata']['y']-1)
    model       = copy.copy(fit['cv_pred'])
    pre_model   = copy.copy(fit['preliminary']['cv_pred'])
    s_model     = copy.copy(fit['session_timing']['cv_pred'])

    if plot_this:
        plt.figure()
        alarms,hits,thresholds = metrics.roc_curve(data,model)
        pre_alarms,pre_hits,pre_thresholds = metrics.roc_curve(data,pre_model)
        s_alarms,s_hits,s_thresholds = metrics.roc_curve(data,s_model)
        plt.plot(alarms,hits,'r-',label='Average')
        plt.plot(pre_alarms,pre_hits,'k-',label='10 Regressors')
        plt.plot(s_alarms,s_hits,'b-',label='Session 1D')
        plt.plot([0,1],[0,1],'k--')
        plt.ylabel('Hits')
        plt.xlabel('False Alarms')
        plt.legend()
    return metrics.roc_auc_score(data,model), metrics.roc_auc_score(data,pre_model), metrics.roc_auc_score(data,s_model)

# UPDATE_REQUIRED
def compare_timing_versions(ids, directory):
    rocs = []
    for id in ids:
        try:
            fit = load_fit(id,directory=directory)
            roc = compute_model_roc_timing(fit)
            rocs.append(roc)
        except:
            pass
    rocs = np.vstack(rocs)
    
    plt.figure()
    plt.plot(rocs.T,'o')
    means =np.mean(rocs,0)
    for i in range(0,3):
        plt.plot([i-0.25,i+0.25],[means[i], means[i]],'k-',linewidth=2)
    plt.ylim([0.5, 1])
    plt.gca().set_xticks([0, 1,2])
    plt.gca().set_xticklabels(['1D Average','10 Timing','1D Session'],{'fontsize':12})
    plt.ylabel('CV ROC')
    
    plt.figure()
    plt.plot(rocs[:,0],rocs[:,1],'o')
    plt.plot([0.5,1],[0.5,1],'k--',alpha=0.3)
    plt.ylabel('CV ROC - Session specific')
    plt.xlabel('CV ROC - Average Timing')
    
    return rocs

# UPDATE_REQUIRED
def summarize_fits(ids, directory):
    crashed = 0
    for id in tqdm(ids):
        try:
            fit = load_fit(id, directory=directory)
            summarize_fit(fit,directory=directory, savefig=True)
        except Exception as e:
            print(e)
            crashed +=1
        plt.close('all')
    print(str(crashed) + " crashed")

def build_model_training_manifest(version=None,verbose=False):
    '''
        Builds a manifest of model results
        Each row is a behavior_session_id
        
        if verbose, logs each crashed session id
        if use_full_ophys, uses the full model for ophys sessions (includes omissions)
    
    '''
    manifest = pgt.get_training_manifest().query('active').copy()
    directory = get_directory(version)

    manifest['behavior_fit_available'] = manifest['active'] #Just copying the column size
    first = True
    crashed = 0
    for index, row in tqdm(manifest.iterrows(),total=manifest.shape[0]):
        try:
            fit = load_fit(row.behavior_session_id,version=version)
        except:
            if verbose:
                print(str(row.behavior_session_id)+" crash")
            manifest.at[index,'behavior_fit_available'] = False
            crashed +=1
        else:
            fit = engagement_for_model_manifest(fit) 
            manifest.at[index,'behavior_fit_available'] = True
            manifest.at[index, 'num_hits']  = np.sum(fit['psydata']['hits'])
            manifest.at[index, 'num_fa']    = np.sum(fit['psydata']['false_alarms'])
            manifest.at[index, 'num_cr']    = np.sum(fit['psydata']['correct_reject'])
            manifest.at[index, 'num_miss']  = np.sum(fit['psydata']['misses'])
            manifest.at[index, 'num_aborts']= np.sum(fit['psydata']['aborts'])
            manifest.at[index, 'fraction_engaged'] = fit['psydata']['full_df']['engaged'].mean() 
            sigma = fit['hyp']['sigma']
            wMode = fit['wMode']
            weights = get_weights_list(fit['weights'])
            manifest.at[index,'session_roc'] = compute_model_roc(fit)
            manifest.at[index,'lick_fraction']          = get_lick_fraction(fit)
            manifest.at[index,'lick_fraction_1st_half'] = get_lick_fraction(fit,first_half=True)
            manifest.at[index,'lick_fraction_2nd_half'] = get_lick_fraction(fit,second_half=True)
            manifest.at[index,'lick_hit_fraction']          = get_hit_fraction(fit)
            manifest.at[index,'lick_hit_fraction_1st_half'] = get_hit_fraction(fit,first_half=True)
            manifest.at[index,'lick_hit_fraction_2nd_half'] = get_hit_fraction(fit,second_half=True)
            manifest.at[index,'trial_hit_fraction']          = get_trial_hit_fraction(fit)
            manifest.at[index,'trial_hit_fraction_1st_half'] = get_trial_hit_fraction(fit,first_half=True)
            manifest.at[index,'trial_hit_fraction_2nd_half'] = get_trial_hit_fraction(fit,second_half=True)

            model_dex, taskdex,timingdex = get_timing_index_fit(fit,return_all=True)
            manifest.at[index,'strategy_dropout_index'] = model_dex
            manifest.at[index,'visual_only_dropout_index'] = taskdex
            manifest.at[index,'timing_only_dropout_index'] = timingdex

            if first:
                possible_weights = {'bias','task0','timing1D','omissions','omissions1'}
                for weight in possible_weights: 
                    manifest['weight_'+weight] = [[]]*len(manifest)
                first=False 

            for dex, weight in enumerate(weights):
                manifest.at[index, 'prior_'+weight] =sigma[dex]
                manifest.at[index, 'avg_weight_'+weight] = np.mean(wMode[dex,:])
                manifest.at[index, 'avg_weight_'+weight+'_1st_half'] = np.mean(wMode[dex,fit['psydata']['flash_ids']<2400])
                if len(fit['psydata']['flash_ids']) >=2400:
                    manifest.at[index, 'avg_weight_'+weight+'_2nd_half'] = np.mean(wMode[dex,fit['psydata']['flash_ids']>=2400])
                manifest.at[index, 'weight_'+str(weight)] = wMode[dex,:]  

    manifest = manifest.query('behavior_fit_available').copy()
    manifest['strategy_weight_index']           = manifest['avg_weight_task0'] - manifest['avg_weight_timing1D']
    manifest['strategy_weight_index_1st_half']  = manifest['avg_weight_task0_1st_half'] - manifest['avg_weight_timing1D_1st_half']
    manifest['strategy_weight_index_2nd_half']  = manifest['avg_weight_task0_2nd_half'] - manifest['avg_weight_timing1D_2nd_half']
    manifest['visual_strategy_session']         = -manifest['visual_only_dropout_index'] > -manifest['timing_only_dropout_index']

    n = len(manifest)
    print(str(crashed)+ " sessions crashed")
    print(str(n) + " sessions returned")
    
    return manifest

def build_model_manifest(version=None,container_in_order=False, full_active_container=False,verbose=False):
    '''
        Builds a manifest of model results
        Each row is a Behavior_session_id
        
        if container_in_order, then only returns sessions that come from a container that was collected in order. The container
            does not need to be complete, as long as the sessions that are present were collected in order
        if full_active_container, then only returns sessions that come from a container with 4 active sessions. 
        if verbose, logs each crashed session id
    
    '''
    manifest = pgt.get_ophys_manifest().query('active').copy()
    directory=get_directory(version) 

    manifest['behavior_fit_available'] = manifest['trained_A'] #Just copying the column size
    first = True
    crashed = 0
    for index, row in tqdm(manifest.iterrows(),total=manifest.shape[0]):
        try:
            fit = load_fit(row.behavior_session_id,version=version)
        except:
            if verbose:
                print(str(row.behavior_session_id)+" crash")
            manifest.at[index,'behavior_fit_available'] = False
            crashed +=1
        else:
            fit = engagement_for_model_manifest(fit) 
            manifest.at[index,'behavior_fit_available'] = True
            manifest.at[index, 'num_hits'] = np.sum(fit['psydata']['hits'])
            manifest.at[index, 'num_fa'] = np.sum(fit['psydata']['false_alarms'])
            manifest.at[index, 'num_cr'] = np.sum(fit['psydata']['correct_reject'])
            manifest.at[index, 'num_miss'] = np.sum(fit['psydata']['misses'])
            manifest.at[index, 'num_aborts'] = np.sum(fit['psydata']['aborts'])
            manifest.at[index, 'fraction_engaged'] = fit['psydata']['full_df']['engaged'].mean() 
            sigma = fit['hyp']['sigma']
            wMode = fit['wMode']
            weights = get_weights_list(fit['weights'])
            manifest.at[index,'session_roc'] = compute_model_roc(fit)
            manifest.at[index,'lick_fraction'] = get_lick_fraction(fit)
            #manifest.at[index,'lick_fraction_1st_half'] = get_lick_fraction(fit,first_half=True)
            #manifest.at[index,'lick_fraction_2nd_half'] = get_lick_fraction(fit,second_half=True)
            manifest.at[index,'lick_hit_fraction'] = get_hit_fraction(fit)
            #manifest.at[index,'lick_hit_fraction_1st_half'] = get_hit_fraction(fit,first_half=True)
            #manifest.at[index,'lick_hit_fraction_2nd_half'] = get_hit_fraction(fit,second_half=True)
            manifest.at[index,'trial_hit_fraction'] = get_trial_hit_fraction(fit)
            #manifest.at[index,'trial_hit_fraction_1st_half'] = get_trial_hit_fraction(fit,first_half=True)
            #manifest.at[index,'trial_hit_fraction_2nd_half'] = get_trial_hit_fraction(fit,second_half=True)
   
            model_dex, taskdex,timingdex = get_timing_index_fit(fit,return_all=True)
            manifest.at[index,'strategy_dropout_index'] = model_dex
            manifest.at[index,'visual_only_dropout_index'] = taskdex
            manifest.at[index,'timing_only_dropout_index'] = timingdex

            dropout_dict = get_session_dropout(fit)
            for dex, weight in enumerate(weights):
                manifest.at[index, 'prior_'+weight] =sigma[dex]
                manifest.at[index, 'dropout_'+weight] = dropout_dict[weight]
                manifest.at[index, 'avg_weight_'+weight] = np.mean(wMode[dex,:])
                #manifest.at[index, 'avg_weight_'+weight+'_1st_half'] = np.mean(wMode[dex,fit['psydata']['flash_ids']<2400])
                #manifest.at[index, 'avg_weight_'+weight+'_2nd_half'] = np.mean(wMode[dex,fit['psydata']['flash_ids']>=2400])
                if first: 
                    manifest['weight_'+weight] = [[]]*len(manifest)
                manifest.at[index, 'weight_'+str(weight)] = wMode[dex,:]  
            first = False
    print(str(crashed)+ " sessions crashed")

    manifest = manifest.query('behavior_fit_available').copy()
    manifest['strategy_weight_index']           = manifest['avg_weight_task0'] - manifest['avg_weight_timing1D']
    #manifest['strategy_weight_index_1st_half']  = manifest['avg_weight_task0_1st_half'] - manifest['avg_weight_timing1D_1st_half']
    #manifest['strategy_weight_index_2nd_half']  = manifest['avg_weight_task0_2nd_half'] - manifest['avg_weight_timing1D_2nd_half']
    manifest['visual_strategy_session']         = -manifest['visual_only_dropout_index'] > -manifest['timing_only_dropout_index']

    # Annotate containers
    in_order = []
    four_active = []
    for index, mouse in enumerate(manifest['container_id'].unique()):
        this_df = manifest.query('container_id == @mouse')
        stages = this_df.session_number.values
        if np.all(stages ==sorted(stages)):
            in_order.append(mouse)
        if len(this_df) == 4:
            four_active.append(mouse)
    manifest['container_in_order'] = manifest.apply(lambda x: x['container_id'] in in_order, axis=1)
    manifest['full_active_container'] = manifest.apply(lambda x: x['container_id'] in four_active,axis=1)

    # Filter and report outcomes
    if container_in_order:
        n_remove = len(manifest.query('not container_in_order'))
        print(str(n_remove) + " sessions out of order")
        manifest = manifest.query('container_in_order')
    if full_active_container:
        n_remove = len(manifest.query('not full_active_container'))
        print(str(n_remove) + " sessions from incomplete active containers")
        manifest = manifest.query('full_active_container')
        if not (np.mod(len(manifest),4) == 0):
            raise Exception('Filtered for full containers, but dont seem to have the right number')
    n = len(manifest)
    print(str(n) + " sessions returned")
    
    return manifest

def engagement_for_model_manifest(fit, lick_threshold=0.1, reward_threshold=1/90, use_bouts=True,win_dur=320, win_type='triang'):
    fit['psydata']['full_df']['bout_rate'] = fit['psydata']['full_df']['bout_start'].rolling(win_dur,min_periods=1, win_type=win_type).mean()/.75
    #fit['psydata']['full_df']['high_lick'] = [True if x > lick_threshold else False for x in fit['psydata']['full_df']['bout_rate']] 
    fit['psydata']['full_df']['reward_rate'] = fit['psydata']['full_df']['hits'].rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75
    #fit['psydata']['full_df']['high_reward'] = [True if x > reward_threshold else False for x in fit['psydata']['full_df']['reward_rate']] 
    #fit['psydata']['full_df']['flash_metrics_epochs'] = [0 if (not x[0]) & (not x[1]) else 1 if x[1] else 2 for x in zip(fit['psydata']['full_df']['high_lick'], fit['psydata']['full_df']['high_reward'])]
    #fit['psydata']['full_df']['flash_metrics_labels'] = ['low-lick,low-reward' if x==0  else 'high-lick,high-reward' if x==1 else 'high-lick,low-reward' for x in fit['psydata']['full_df']['flash_metrics_epochs']]
    #fit['psydata']['full_df']['engaged'] = [(x=='high-lick,low-reward') or (x=='high-lick,high-reward') for x in fit['psydata']['full_df']['flash_metrics_labels']]
    fit['psydata']['full_df']['engaged'] = [x > reward_threshold for x in fit['psydata']['full_df']['reward_rate']]
    return fit



def plot_all_manifest_by_stage(manifest, version,savefig=True, group_label='all'):
    plot_manifest_by_stage(manifest,'session_roc',hline=0.5,ylims=[0.5,1],version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'lick_fraction',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'lick_hit_fraction',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'trial_hit_fraction',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'strategy_dropout_index',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'strategy_weight_index',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'prior_bias',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'prior_task0',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'prior_omissions1',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'prior_timing1D',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'avg_weight_bias',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'avg_weight_task0',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'avg_weight_omissions1',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_stage(manifest,'avg_weight_timing1D',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_stage(manifest,'avg_weight_task0_1st_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_stage(manifest,'avg_weight_task0_2nd_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_stage(manifest,'avg_weight_timing1D_1st_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_stage(manifest,'avg_weight_timing1D_2nd_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_stage(manifest,'avg_weight_bias_1st_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_stage(manifest,'avg_weight_bias_2nd_half',version=version,savefig=savefig,group_label=group_label)

def plot_all_manifest_by_cre(manifest, version,savefig=True, group_label='all'):
    plot_manifest_by_cre(manifest,'session_roc',hline=0.5,ylims=[0.5,1],version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'lick_fraction',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'lick_hit_fraction',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'trial_hit_fraction',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'strategy_dropout_index',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'strategy_weight_index',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'prior_bias',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'prior_task0',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'prior_omissions1',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'prior_timing1D',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'avg_weight_bias',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'avg_weight_task0',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'avg_weight_omissions1',version=version,savefig=savefig,group_label=group_label)
    plot_manifest_by_cre(manifest,'avg_weight_timing1D',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_cre(manifest,'avg_weight_task0_1st_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_cre(manifest,'avg_weight_task0_2nd_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_cre(manifest,'avg_weight_timing1D_1st_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_cre(manifest,'avg_weight_timing1D_2nd_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_cre(manifest,'avg_weight_bias_1st_half',version=version,savefig=savefig,group_label=group_label)
    #plot_manifest_by_cre(manifest,'avg_weight_bias_2nd_half',version=version,savefig=savefig,group_label=group_label)

def compare_all_manifest_by_stage(manifest, version, savefig=True, group_label='all'):
    compare_manifest_by_stage(manifest,['3','4'], 'strategy_weight_index',version=version,savefig=savefig,group_label=group_label)
    compare_manifest_by_stage(manifest,['3','4'], 'strategy_dropout_index',version=version,savefig=savefig,group_label=group_label)    
    compare_manifest_by_stage(manifest,['3','4'], 'avg_weight_task0',version=version,savefig=savefig,group_label=group_label)
    compare_manifest_by_stage(manifest,['3','4'], 'avg_weight_timing1D',version=version,savefig=savefig,group_label=group_label)
    compare_manifest_by_stage(manifest,['3','4'], 'session_roc',version=version,savefig=savefig,group_label=group_label)

def get_clean_session_names(session_numbers):
    names = {
        1:'F1',
        2:'F2',
        3:'F3',
        4:'N1',
        5:'N2',
        6:'N3',
        '1':'F1',
        '2':'F2',
        '3':'F3',
        '4':'N1',
        '5':'N2',
        '6':'N3'}

    return np.array([names[x] for x in session_numbers])

def plot_manifest_by_stage(manifest, key,ylims=None,hline=0,version=None,savefig=True,group_label='all',stage_names=None,fs1=12,fs2=12,filetype='.png',force_fig_size=None):
    means = manifest.groupby('session_number')[key].mean()
    sem = manifest.groupby('session_number')[key].sem()
    if stage_names is None:
        stage_names = np.array(manifest.groupby('session_number')[key].mean().index) 
    clean_names = get_clean_session_names(stage_names)
    if type(force_fig_size) == type(None):
        plt.figure()
    else:
        plt.figure(figsize=force_fig_size)
    #colors = sns.color_palette("hls",len(means))
    colors = pstyle.get_project_colors(keys=clean_names)
    for index, m in enumerate(means):
        plt.plot([index-0.5,index+0.5], [m, m],'-',color=colors[clean_names[index]],linewidth=4)
        plt.plot([index, index],[m-sem.iloc[index], m+sem.iloc[index]],'-',color=colors[clean_names[index]])

    plt.gca().set_xticks(np.arange(0,len(stage_names)))
    plt.gca().set_xticklabels(clean_names,rotation=0,fontsize=fs1)
    plt.gca().axhline(hline, alpha=0.3,color='k',linestyle='--')
    plt.yticks(fontsize=fs2)
    plt.ylabel(key,fontsize=fs1)
    stage3, stage4 = get_manifest_values_by_stage(manifest, ['3','4'],key)
    pval =  ttest_rel(stage3,stage4,nan_policy='omit')
    ylim = plt.ylim()[1]
    plt.plot([1,2],[ylim*1.05, ylim*1.05],'k-')
    plt.plot([1,1],[ylim, ylim*1.05], 'k-')
    plt.plot([2,2],[ylim, ylim*1.05], 'k-')

    if pval[1] < 0.05:
        plt.plot(1.5, ylim*1.1,'k*')
    else:
        plt.text(1.5,ylim*1.1, 'ns')
    if ylims is not None:
        plt.ylim(ylims)
    plt.tight_layout()    

    if savefig:
        directory=get_directory(version)
        plt.savefig(directory+'figures_summary/'+group_label+"_stage_comparisons_"+key+filetype)

def get_manifest_values_by_cre(manifest,cres, key):
    x = cres[0] 
    y = cres[1]
    z = cres[2]
    s1df = manifest.query('cre_line ==@x')[key].drop_duplicates(keep='last')
    s2df = manifest.query('cre_line ==@y')[key].drop_duplicates(keep='last')
    s3df = manifest.query('cre_line ==@z')[key].drop_duplicates(keep='last')
    return s1df.values, s2df.values, s3df.values 

def get_manifest_values_by_stage(manifest, stages, key):
    x = stages[0]
    y = stages[1]
    s1df = manifest.set_index(['container_id']).query('session_number ==@x')[key].drop_duplicates(keep='last')
    s2df = manifest.set_index(['container_id']).query('session_number ==@y')[key].drop_duplicates(keep='last')
    s1df.name=x
    s2df.name=y
    full_df = s1df.to_frame().join(s2df)
    vals1 = full_df[x].values 
    vals2 = full_df[y].values 
    return vals1,vals2  

def compare_manifest_by_stage(manifest,stages, key,version=None,savefig=True,group_label='all'):
    '''
        Function for plotting various metrics by ophys_stage
        compare_manifest_by_stage(manifest,['1','3'],'avg_weight_task0')
    '''
    directory=get_directory(version)
    # Get the stage values paired by container
    vals1, vals2 = get_manifest_values_by_stage(manifest, stages, key)

    plt.figure(figsize=(6,5))
    plt.plot(vals1,vals2,'ko')
    xlims = plt.xlim()
    ylims = plt.ylim()
    all_lims = np.concatenate([xlims,ylims])
    lims = [np.min(all_lims), np.max(all_lims)]
    plt.plot(lims,lims, 'k--')
    stage_names = get_clean_session_names(stages)
    plt.xlabel(stage_names[0],fontsize=12)
    plt.ylabel(stage_names[1],fontsize=12)
    plt.title(key)
    pval = ttest_rel(vals1,vals2,nan_policy='omit')
    ylim = plt.ylim()[1]
    if pval[1] < 0.05:
        plt.title(key+": *")
    else:
        plt.title(key+": ns")
    plt.tight_layout()    

    if savefig:
        plt.savefig(directory+'figures_summary/'+group_label+"_stage_comparisons_"+stages[0]+"_"+stages[1]+"_"+key+".png")

def plot_static_comparison(IDS, version=None,savefig=False,group_label=""):
    '''
        Top Level function for comparing static and dynamic logistic regression using ROC scores
    '''

    directory=get_directory(version)

    all_s, all_d = get_all_static_comparisons(IDS, version)
    plot_static_comparison_inner(all_s,all_d,version=version, savefig=savefig, group_label=group_label)

def plot_static_comparison_inner(all_s,all_d,version=None, savefig=False,group_label="",fs1=12,fs2=12,filetype='.png'): 
    '''
        Plots static and dynamic ROC comparisons
    
    '''
    fig,ax = plt.subplots(figsize=(5,4))
    plt.plot(all_s,all_d,'ko')
    plt.plot([0.5,1],[0.5,1],'k--')
    plt.ylabel('Dynamic ROC',fontsize=fs1)
    plt.xlabel('Static ROC',fontsize=fs1)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.tight_layout()
    if savefig:
        directory=get_directory(version)
        plt.savefig(directory+"figures_summary/summary_static_comparison"+group_label+filetype)

def get_all_static_comparisons(IDS, version):
    '''
        Iterates through list of session ids and gets static and dynamic ROC scores
    '''
    all_s = []
    all_d = []    

    for index, id in enumerate(IDS):
        try:
            fit = load_fit(id, version=version)
            static,dynamic = get_static_roc(fit)
        except:
            pass
        else:
            all_s.append(static)
            all_d.append(dynamic)

    return all_s, all_d

def get_static_design_matrix(fit):
    '''
        Returns the design matrix to be used for static logistic regression, does not include bias
    '''
    X = []
    for index, w in enumerate(fit['weights'].keys()):
        if fit['weights'][w]:
            if not (w=='bias'):
                X.append(fit['psydata']['inputs'][w]) 
    return np.hstack(X)

def get_static_roc(fit,use_cv=False):
    '''
        Returns the area under the ROC curve for a static logistic regression model
    '''
    X = get_static_design_matrix(fit)
    y = fit['psydata']['y'] - 1
    if use_cv:
        clf = logregcv(cv=10)
    else:
        clf = logreg(penalty='none',solver='lbfgs')
    clf.fit(X,y)
    ypred = clf.predict(X)
    fpr, tpr, thresholds = metrics.roc_curve(y,ypred)
    static_roc = metrics.auc(fpr,tpr)
    dfpr, dtpr, dthresholds = metrics.roc_curve(y,fit['cv_pred'])
    dynamic_roc = metrics.auc(dfpr,dtpr)   
    return static_roc, dynamic_roc

def plot_manifest_by_cre(manifest,key,ylims=None,hline=0,version=None,savefig=True,group_label='all',fs1=12,fs2=12,rotation=0,labels=None,figsize=None,ylabel=None):
    means = manifest.groupby('cre_line')[key].mean()
    sem  = manifest.groupby('cre_line')[key].sem()
    if figsize is None:
        plt.figure()
    else:
        plt.figure(figsize=figsize)
    if labels is None:
        names = np.array(manifest.groupby('cre_line')[key].mean().index) 
    else:
        names = labels
    colors = pstyle.get_project_colors()
    for index, m in enumerate(means):
        plt.plot([index-0.5,index+0.5], [m, m],'-',color=colors[names[index]],linewidth=4)
        plt.plot([index, index],[m-sem.iloc[index], m+sem.iloc[index]],'-',color=colors[names[index]])

    plt.gca().set_xticks(np.arange(0,len(names)))
    plt.gca().set_xticklabels(names,rotation=rotation,fontsize=fs1)
    plt.gca().axhline(hline, alpha=0.3,color='k',linestyle='--')
    plt.yticks(fontsize=fs2)
    if ylabel is None:
        plt.ylabel(key,fontsize=fs1)
    else:
        plt.ylabel(ylabel,fontsize=fs1)
    cres = means.index.values
    c1,c2,c3 = get_manifest_values_by_cre(manifest,cres,key)
    pval12 =  ttest_ind(c1,c2,nan_policy='omit')
    pval13 =  ttest_ind(c1,c3,nan_policy='omit')
    pval23 =  ttest_ind(c2,c3,nan_policy='omit')
    ylim = plt.ylim()[1]
    r = plt.ylim()[1] - plt.ylim()[0]
    sf = .075
    offset = 2 
    plt.plot([0,1],[ylim+r*sf, ylim+r*sf],'k-')
    plt.plot([0,0],[ylim, ylim+r*sf], 'k-')
    plt.plot([1,1],[ylim, ylim+r*sf], 'k-')
 
    plt.plot([0,2],[ylim+r*sf*3, ylim+r*sf*3],'k-')
    plt.plot([0,0],[ylim+r*sf*2, ylim+r*sf*3], 'k-')
    plt.plot([2,2],[ylim+r*sf*2, ylim+r*sf*3], 'k-')

    plt.plot([1,2],[ylim+r*sf, ylim+r*sf],'k-')
    plt.plot([1,1],[ylim, ylim+r*sf], 'k-')
    plt.plot([2,2],[ylim, ylim+r*sf], 'k-')

    if pval12[1] < 0.05:
        plt.plot(.5, ylim+r*sf*1.5,'k*')
    else:
        plt.text(.5,ylim+r*sf*1.25, 'ns')

    if pval13[1] < 0.05:
        plt.plot(1, ylim+r*sf*3.5,'k*')
    else:
        plt.text(1,ylim+r*sf*3.5, 'ns')

    if pval23[1] < 0.05:
        plt.plot(1.5, ylim+r*sf*1.5,'k*')
    else:
        plt.text(1.5,ylim+r*sf*1.25, 'ns')

    if ylims is not None:
        plt.ylim(ylims)
    plt.tight_layout()    

    if savefig:
        directory=get_directory(version)
        plt.savefig(directory+'figures_summary/'+group_label+"_cre_comparisons_"+key+".png")
        plt.savefig(directory+'figures_summary/'+group_label+"_cre_comparisons_"+key+".svg")

def plot_task_index_by_cre(manifest,version=None,savefig=True,group_label='all',strategy_matched=False):
    if strategy_matched:
        manifest = manifest.query('strategy_matched').copy()
        group_label=group_label+'_strategy_matched'
    directory=get_directory(version)
    plt.figure(figsize=(5,4))
    #cre = manifest.cre_line.unique()
    cre = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
    colors = pstyle.get_project_colors(keys=cre)
    for i in range(0,len(cre)):
        x = manifest.cre_line.unique()[i]
        df = manifest.query('cre_line == @x')
        plt.plot(-df['visual_only_dropout_index'], -df['timing_only_dropout_index'], 'o',color=colors[x],label=x,alpha=1)
    plt.plot([0,40],[0,40],'k--',alpha=0.5)
    plt.ylabel('Timing Index',fontsize=20)
    plt.xlabel('Visual Index',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_cre.png")
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_cre.svg")

    plt.figure(figsize=(8,3))
    cre = manifest.cre_line.unique()
    s = 0
    for i in range(0,len(cre)):
        x = manifest.cre_line.unique()[i]
        df = manifest.query('cre_line == @x')
        plt.plot(np.arange(s,s+len(df)), df['strategy_dropout_index'].sort_values(), 'o',color=colors[x],label=x)
        s += len(df)
    plt.axhline(0,ls='--',color='k',alpha=0.5)
    plt.ylabel('Strategy Dropout Index',fontsize=12)
    plt.xlabel('Session',fontsize=12)
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_cre_each_sequence.png")
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_cre_each_sequence.svg")

    plt.figure(figsize=(8,3))
    cre = manifest.cre_line.unique()
    sorted_manifest = manifest.sort_values(by='strategy_dropout_index')
    count = 0
    for index, row in sorted_manifest.iterrows():
        if row.cre_line == cre[0]:
            plt.plot(count, row.strategy_dropout_index, 'o',color=colors[row.cre_line])
        elif row.cre_line == cre[1]:
            plt.plot(count,row.strategy_dropout_index, 'o',color=colors[row.cre_line])
        else:
            plt.plot(count,row.strategy_dropout_index, 'o',color=colors[row.cre_line])
        count+=1
    plt.axhline(0,ls='--',color='k',alpha=0.5)
    plt.ylabel('Strategy Dropout Index',fontsize=12)
    plt.xlabel('Session',fontsize=12)
    plt.tight_layout()

    if savefig:
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_cre_sequence.png")
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_cre_sequence.svg")

    plt.figure(figsize=(5,4))
    counts,edges = np.histogram(manifest['strategy_dropout_index'].values,20)
    plt.axvline(0,ls='--',color='k',alpha=0.5)
    cre = ['Slc17a7-IRES2-Cre','Sst-IRES-Cre','Vip-IRES-Cre']
    for i in range(0,len(cre)):
        x = manifest.cre_line.unique()[i]
        df = manifest.query('cre_line == @x')
        plt.hist(df['strategy_dropout_index'].values, bins=edges,alpha=.5,color=colors[x],label=x)

    plt.ylabel('Count',fontsize=20)
    plt.xlabel('Strategy Dropout Index',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.tight_layout()

    if savefig:
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_cre_histogram.png")
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_cre_histogram.svg")

def plot_manifest_by_date(manifest,version=None,savefig=True,group_label='all',plot_by=4):
    directory=get_directory(version)
    manifest = manifest.sort_values(by=['date_of_acquisition'])
    plt.figure(figsize=(8,4))
    plt.plot(manifest.date_of_acquisition,manifest.strategy_dropout_index,'ko')
    plt.axhline(0,ls='--',color='k',alpha=0.5)
    plt.gca().set_xticks(manifest.date_of_acquisition.values[::plot_by])
    labels = manifest.date_of_acquisition.values[::plot_by]
    labels = [x[0:10] for x in labels]
    plt.gca().set_xticklabels(labels,rotation=-90)
    plt.ylabel('Strategy Dropout Index',fontsize=12)
    plt.xlabel('Date of Acquisition',fontsize=12)
    plt.tight_layout()

    if savefig:
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_date.png")

def plot_task_timing_over_session(manifest,version=None,savefig=True,group_label='all'):
    directory=get_directory(version)
    weight_task_index_by_flash = [manifest.loc[x]['weight_task0'] - manifest.loc[x]['weight_timing1D'] for x in manifest.index]
    wtibf = np.vstack([x[0:3100] for x in weight_task_index_by_flash])
    plt.figure(figsize=(8,3))
    for x in weight_task_index_by_flash:
        plt.plot(x,'k',alpha=0.1)
    plt.plot(np.mean(wtibf,0),linewidth=4)
    plt.axhline(0,ls='--',color='k')
    plt.ylim(-5,5)
    plt.xlim(0,3200)
    plt.ylabel('Strategy Dropout Index',fontsize=12)
    plt.xlabel('Flash # in session',fontsize=12)
    plt.tight_layout()

    if savefig:
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_over_session.png")


def plot_task_timing_by_training_duration(model_manifest,version=None, savefig=True,group_label='all'):
    avg_index = []
    num_train_sess = []
    behavior_sessions = pgt.get_training_manifest()
    behavior_sessions['training'] = behavior_sessions['ophys_session_id'].isnull()
    for index, mouse in enumerate(pgt.get_mice_ids()):
        df = behavior_sessions.query('donor_id ==@mouse')
        num_train_sess.append(len(df.query('training')))
        avg_index.append(model_manifest.query('donor_id==@mouse').strategy_dropout_index.mean())

    plt.figure()
    plt.plot(avg_index, num_train_sess,'ko')
    plt.ylabel('Number of Training Sessions')
    plt.xlabel('Strategy Index')
    plt.axvline(0,ls='--',color='k')
    plt.axhline(0,ls='--',color='k')

    if savefig:
        directory=get_directory(version)
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_train_duration.png")

def clean_keys():
    keys_dict = {
        'dropout_task0':'Visual Dropout',    
        'dropout_timing1D':'Timing Dropout', 
        'dropout_omissions':'Omission Dropout',
        'dropout_omissions1':'Prev. Omission Dropout'
    }
    return keys_dict

def scatter_manifest(model_manifest, key1, key2, version=None,sflip1=False,sflip2=False,cindex=None, savefig=True,group_label='all',plot_regression=False):
    directory=get_directory(version)
    vals1 = model_manifest[key1].values
    vals2 = model_manifest[key2].values
    keys_dict = clean_keys()
    if sflip1:
        vals1 = -vals1
    if sflip2:
        vals2 = -vals2
    plt.figure()
    if (type(cindex) == type(None)):
       plt.plot(vals1,vals2,'ko')
    else:
        ax = plt.gca()
        scat = ax.scatter(vals1,vals2,c=model_manifest[cindex],cmap='plasma')
        cbar = plt.gcf().colorbar(scat, ax = ax)
        cbar.ax.set_ylabel(cindex,fontsize=12)
    plt.xlabel(keys_dict.get(key1,key1),fontsize=16)
    plt.ylabel(keys_dict.get(key2,key2),fontsize=16)
    plt.gca().xaxis.set_tick_params(labelsize=16)
    plt.gca().yaxis.set_tick_params(labelsize=16)

    if plot_regression:    
        x = np.array(vals1).reshape((-1,1))
        y = np.array(vals2)
        model = LinearRegression(fit_intercept=False).fit(x,y)
        sortx = np.sort(x).reshape((-1,1))
        y_pred = model.predict(sortx)
        plt.plot(sortx,y_pred, 'r--')
        score = round(model.score(x,y),2)
        #plt.text(sortx[0],y_pred[-1],"Omissions = "+str(round(model.coef_[0],2))+"*Task \nr^2 = "+str(score),color="r",fontsize=16)

    if savefig:
        if (type(cindex) == type(None)):
            plt.savefig(directory+'figures_summary/'+group_label+"_manifest_scatter_"+key1+"_by_"+key2+".png")
        else:
            plt.savefig(directory+'figures_summary/'+group_label+"_manifest_scatter_"+key1+"_by_"+key2+"_with_"+cindex+"_colorbar.png")

def plot_manifest_groupby(manifest, key, group, savefig=True, version=None, group_label='all'):
    directory = get_directory(version)
    means = manifest.groupby(group)[key].mean()
    sem = manifest.groupby(group)[key].sem()
    names = np.array(manifest.groupby(group)[key].mean().index) 
    plt.figure()
    colors = sns.color_palette("hls",len(means))
    #colors = pstyle.get_project_colors(names)
    for index, m in enumerate(means):
        plt.plot([index-0.5,index+0.5], [m, m],'-',color=colors[index],linewidth=4)
        plt.plot([index, index],[m-sem.iloc[index], m+sem.iloc[index]],'-',color=colors[index])

    plt.gca().set_xticks(np.arange(0,len(names)))
    plt.gca().set_xticklabels(names,rotation=0,fontsize=12)
    plt.gca().axhline(0, alpha=0.3,color='k',linestyle='--')
    plt.ylabel(key,fontsize=12)
    plt.xlabel(group, fontsize=12)

    if len(means) == 2:
        # Do significance testing 
        groups = manifest.groupby(group)
        vals = []
        for name, grouped in groups:
            vals.append(grouped[key])
        pval =  ttest_ind(vals[0],vals[1],nan_policy='omit')
        ylim = plt.ylim()[1]
        r = plt.ylim()[1] - plt.ylim()[0]
        sf = .075
        offset = 2 
        plt.plot([0,1],[ylim+r*sf, ylim+r*sf],'k-')
        plt.plot([0,0],[ylim, ylim+r*sf], 'k-')
        plt.plot([1,1],[ylim, ylim+r*sf], 'k-')
     
        if pval[1] < 0.05:
            plt.plot(.5, ylim+r*sf*1.5,'k*')
        else:
            plt.text(.5,ylim+r*sf*1.25, 'ns')

    if savefig:
        plt.savefig(directory+'figures_summary/'+group_label+"_manifest_"+key+"_groupby_"+group+".png")


def omissions_by_exposure(ophys,maxval=4,metric='weight'):
    plt.figure()
    colors = plt.get_cmap('tab20c')
    
    if metric == 'weight':
        visual_metric='visual_weight_index_engaged'
        omission_metric='omissions1_weight_index_engaged'
    else:
        visual_metric='dropout_task0'
        omission_metric='dropout_omission1'   

    for i in range(0,maxval):
        temp = ophys.query('session_number in [1,3]').query('prior_exposures_to_omissions == @i')
        if metric =='weight':
            plt.plot(temp[visual_metric], temp[omission_metric], 'o', color =colors(i), label=str(i))
        else:
            plt.plot(-temp[visual_metric], -temp[omission_metric], 'o', color =colors(i), label=str(i))   
    for i in range(maxval,2*maxval):
        temp = ophys.query('session_number in [4,6]').query('prior_exposures_to_omissions == @i')
        if metric =='weight':
            plt.plot(temp[visual_metric], temp[omission_metric], 'o', color =colors(i), label=str(i)+' novel')
        else:
            plt.plot(-temp[visual_metric], -temp[omission_metric], 'o', color =colors(i), label=str(i)+' novel')   
    
    plt.legend()
