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
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import k_means
from sklearn.decomposition import PCA

import psy_style as pstyle
import psy_metrics_tools as pm
import psy_general_tools as pgt

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
    if type(version) is str:
        version = int(version)
    directory = pgt.get_directory(version, verbose=True,subdirectory='fits')
    filename = directory + str(bsid)
    fig_dir = pgt.get_directory(version, subdirectory='session_figures')
    fig_filename = fig_dir +str(bsid)
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
    format_options = get_format_options(version, format_options)
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
        # TODO, Issue #188
        fit = cluster_fit(fit,directory=pgt.get_directory(version, subdirectory='clusters')) # gets saved separately

    print('Saving fit dictionary')
    save(filename+".pkl", fit) 
    summarize_fit(fit, version=20, savefig=True)
    plt.close('all')

    print('Saving strategy df')
    build_session_strategy_df(bsid, version,fit=fit,session=session)

    print('Saving licks df')
    build_session_licks_df(session, bsid, version)


def build_session_strategy_df(bsid, version,TRAIN=False,fit=None,session=None):
    '''
        Saves an analysis file in <output_dir> for the model fit of session <id> 
        Extends model weights to be constant during licking bouts

        licked (bool) Did the mouse lick during this image?
        lick_bout_start (bool) did the mouse start a lick bout during this image?
        lick_bout_end (bool) did a lick bout end during this image?
        lick_rate (float) ?? #TODO #200
        in_lick_bout (bool) 
        lick_bout_rate ( float) ?? #TODO #200
        rewarded (bool) did the mouse get a reward during this image?
        lick_hit_fraction (float) ?? #TODO #200
        hit_rate (float) ?? #TODO #200
        miss_rate           #TODO #200
        false_alarm_rate    #TODO #200
        correct_reject_rate #TODO #200
        d_prime             #TODO #200
        criterion           #TODO #200
        RT                  #TODO #200
        engaged             #TODO #200
        strategy weights
            bias
            omissions
            omissions1
            task0
            timing1D
    '''
    # Get Stimulus Info, append model free metrics
    if session is None:
        session = pgt.get_data(bsid)
        pm.get_metrics(session)
    else:
        # add checks here to see if it has already been added?
        if 'bout_number' not in session.licks:
            pm.annotate_licks(session)
        if 'bout_start' not in session.stimulus_presentations:
            pm.annotate_bouts(session)
        if 'reward_rate' not in session.stimulus_presentations:
            pm.annotate_flash_rolling_metrics(session)

    # Load Model fit
    if fit is None:
        fit = load_fit(bsid, version=version)
 
    # include when licking bout happened
    session.stimulus_presentations['in_lick_bout'] = fit['psydata']['full_df']['in_bout'].astype(bool)
 
    # include model weights
    weights = get_weights_list(fit['weights'])
    for wdex, weight in enumerate(weights):
        session.stimulus_presentations.at[~session.stimulus_presentations['in_lick_bout'].values, weight] = fit['wMode'][wdex,:]

    # Iterate value from start of bout forward
    session.stimulus_presentations.fillna(method='ffill', inplace=True)

    # Clean up Stimulus Presentations
    model_output = session.stimulus_presentations.copy()
    model_output.drop(columns=['duration', 'end_frame', 'image_set','index', 
        'orientation', 'start_frame', 'start_time', 'stop_time', 'licks', 
        'rewards', 'time_from_last_lick', 'time_from_last_reward', 
        'time_from_last_change', 'mean_running_speed', 'num_bout_start', 
        'num_bout_end','change_with_lick','change_without_lick',
        'non_change_with_lick','non_change_without_lick','hit_bout'
        ],inplace=True,errors='ignore') 

    # Clean up some names created in psy_metrics
    model_output = model_output.rename(columns={
        'bout_end':'lick_bout_end', 
        'bout_start':'lick_bout_start',
        'bout_rate':'lick_bout_rate'
        })

    # Save out dataframe
    model_output.to_csv(pgt.get_directory(version, subdirectory='strategy_df')+str(bsid)+'.csv') 


def build_session_licks_df(session, bsid, version):
    '''
        Saves a dataframe of the lick times for this session
    
        timestamps  (float) time of lick
        pre_ili (float) time from last lick
        post_ili (float) time until next lick
        rewarded (bool) whether this lick was rewarded
        bout_start (bool) whether this lick was the start of a licking bout
        bout_end (bool) whether this lick was the end of a licking bout
        bout_number (bool) oridinal numbering of bouts in this session
        bout_rewarded (bool) whether this licking bout was rewarded
        behavior_session_id (int64) 
    '''

    # Annotate licks if missing
    if 'bout_number' not in session.licks:
        pm.annotate_licks(session)

    # Grab licks df
    session_licks_df = session.licks

    # Save out dataframe
    filename = pgt.get_directory(version, subdirectory='licks_df')+str(bsid)+'.csv'
    session_licks_df.to_csv(filename,index=False) 

 
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
    session.stimulus_presentations['hits']   =  session.stimulus_presentations['licked'] & session.stimulus_presentations['is_change']
    session.stimulus_presentations['misses'] = ~session.stimulus_presentations['licked'] & session.stimulus_presentations['is_change']
    session.stimulus_presentations['aborts'] =  session.stimulus_presentations['licked'] & ~session.stimulus_presentations['is_change']
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


def get_format_options(version, format_options):
    '''
        Defines the default format options, and sets any values not passed in
    '''
    print('Loading options for version '+str(version))
    defaults = pgt.load_version_parameters(version)

    for k in defaults.keys():
        if k not in format_options:
            format_options[k] = defaults[k]
        else:
            print('Overriding default parameter: '+k)

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
        # TODO Issue, #211
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
    # TODO, this is probably outdated, right? Issue #138
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
    # TODO Issue, #211
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
    weights_list = pgt.get_clean_string(get_weights_list(weights))
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
    # TODO, Issue #188
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
        print("\rrunning fold " +str(k),end="") # TODO, update with tqdm
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


def compute_model_roc(fit,plot_this=False,cross_validation=True):
    '''
        Computes area under the ROC curve for the model in fit. 
        
        plot_this (bool), plots the ROC curve. 
        cross_validation (bool)
            if True uses the cross validated prediction in fit
            if False uses the training fit

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

def load_fit(bsid, version=None):
    '''
        Loads the fit for session bsid, in directory
        Creates a dictionary for the session
        if the fit has cluster labels then it loads them and puts them into the dictionary
    '''
    directory = pgt.get_directory(version,subdirectory='fits')
    filename = directory + str(bsid) + ".pkl" 
    output = load(filename)
    if type(output) is not dict:
        labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
        fit = dict((x,y) for x,y in zip(labels, output))
    else:
        fit = output
    fit['bsid'] = bsid
    # TODO, Issue #188
    if os.path.isfile(directory+str(bsid) + "_all_clusters.pkl"): # probably broken
        fit['all_clusters'] = load(directory+str(bsid) + "_all_clusters.pkl")
    return fit


def load_session_strategy_df(bsid, version, TRAIN=False):
    if TRAIN:
        raise Exception('need to implement')
    else:
        return pd.read_csv(pgt.get_directory(version, subdirectory='strategy_df')+str(bsid)+'.csv') 


def load_session_licks_df(bsid, version):
    licks=pd.read_csv(pgt.get_directory(version,subdirectory='licks_df')+str(bsid)+'.csv')
    licks['behavior_session_id'] = bsid
    return licks


# TODO, Issue #188
def plot_cluster(ID, cluster, fit=None, directory=None):
    if directory is None:
        directory = global_directory
    if type(fit) is not dict: 
        fit = load_fit(ID, directory=directory)
    plot_fit(ID,fit=fit, cluster_labels=fit['clusters'][str(cluster)][1])


def summarize_fit(fit, version=None, savefig=False):
    directory = pgt.get_directory(version)

    fig,ax = plt.subplots(nrows=2,ncols=2, figsize=(10,7))
    my_colors = sns.color_palette("hls",len(fit['weights'].keys()))

    # Plot average weight
    means = np.mean(fit['wMode'],1)
    stds = np.std(fit['wMode'],1)
    weights_list = pgt.get_clean_string(get_weights_list(fit['weights']))
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
    fig.text(.7,starty-offset*0,"Session:  "   ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*0,str(fit['ID']),fontsize=fs)

    fig.text(.7,starty-offset*1,"Driver Line:  " ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*1,fit['metadata']['driver_line'][-1],fontsize=fs)

    fig.text(.7,starty-offset*2,"Stage:  "     ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*2,str(fit['metadata']['session_type']),fontsize=fs)

    fig.text(.7,starty-offset*3,"ROC Train:  ",fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*3,str(round(roc_train,2)),fontsize=fs)

    fig.text(.7,starty-offset*4,"ROC CV:  "    ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*4,str(round(roc_cv,2)),fontsize=fs)

    fig.text(.7,starty-offset*5,"Lick Fraction:  ",fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*5,str(round(fit['psydata']['full_df']['bout_start'].mean(),3)),fontsize=fs)

    fig.text(.7,starty-offset*6,"Lick Hit Fraction:  ",fontsize=fs,horizontalalignment='right')
    lick_hit_fraction = fit['psydata']['full_df']['hits'].sum()/fit['psydata']['full_df']['bout_start'].sum()
    fig.text(.7,starty-offset*6,str(round(lick_hit_fraction,3)),fontsize=fs)

    fig.text(.7,starty-offset*7,"Trial Hit Fraction:  ",fontsize=fs,horizontalalignment='right')
    trial_hit_fraction = fit['psydata']['full_df']['hits'].sum()/fit['psydata']['full_df']['change'].sum()
    fig.text(.7,starty-offset*7,str(round(trial_hit_fraction,3)),fontsize=fs)

    fig.text(.7,starty-offset*8,"Dropout Task/Timing Index:  " ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*8,str(round(get_timing_index_fit(fit),2)),fontsize=fs) 

    fig.text(.7,starty-offset*9,"Weight Task/Timing Index:  " ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*9,str(round(get_weight_timing_index_fit(fit),2)),fontsize=fs)  

    fig.text(.7,starty-offset*10,"Num Hits:  " ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*10,np.sum(fit['psydata']['hits']),fontsize=fs)  

    plt.tight_layout()
    if savefig:
        filename = directory + 'figures_sessions/'+str(fit['ID'])+"_summary.png"
        plt.savefig(filename)
    

def plot_fit(ID, cluster_labels=None,fit=None, version=None,savefig=False,num_clusters=None):
    '''
        Plots the fit associated with a session ID
        Needs the fit dictionary. If you pass these values into, the function is much faster 
    '''

    directory = pgt.get_directory(version)
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
  
# TODO, Issue #188
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

# TODO, Issue #188
def cluster_weights(wMode,num_clusters):
    '''
        Clusters the weights in wMode into num_clusters clusters
    '''
    output = k_means(transform(wMode.T),num_clusters)
    return output

# TODO, Issue #188
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

# TODO, Issue #188
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
    

# TODO, Issue #187
def load_mouse(mouse, get_behavior=False):
    '''
        Takes a mouse donor_id, returns a list of all sessions objects, their IDS, and whether it was active or not. 
        if get_behavior, returns all BehaviorSessions
        no matter what, always returns the behavior_session_id for each session. 
        if global OPHYS, then forces get_behavior=False
    '''
    return pgt.load_mouse(mouse, get_behavior=get_behavior)

# TODO, Issue #187
def format_mouse(sessions,IDS,version, format_options={}):
    '''
        Takes a list of sessions and returns a list of psydata formatted dictionaries for each session, and IDS a list of the IDS that go into each session
    '''
    d =[]
    good_ids =[]
    for session, id in zip(sessions,IDS):
        try:
            pm.annotate_licks(session) 
            pm.annotate_bouts(session)
            format_options = get_format_options(version, format_options)
            psydata = format_session(session,format_options)
        except Exception as e:
            print(str(id) +" "+ str(e))
        else:
            print(str(id))
            d.append(psydata)
            good_ids.append(id)
    return d, good_ids

# TODO, Issue #187
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

# TODO, Issue #187
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

# TODO, Issue #187
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


# TODO, Issue #187
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

# TODO, Issue #187
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

# TODO, Issue #187
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

# TODO, Issue #188
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

# TODO, Issue #188
def merge_weights(w): 
    '''
        Merges a list of weights into one long array of weights
    '''
    return np.concatenate(w,axis=1)           

# TODO, Issue #188
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

# TODO, Issue #187
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

# TODO, Issue #188
def save_session_clusters(session_clusters, directory=None):
    '''
        Saves the session_clusters in 'session_clusters,pkl'

    '''
    if type(directory) == type(None):
        directory = global_directory

    filename = directory + "session_clusters.pkl"
    save(filename,session_clusters)

# TODO, Issue #188
def save_all_clusters(w_ids,session_clusters, directory=None):
    '''
        Saves each sessions all_clusters
    '''
    if type(directory) == type(None):
        directory = global_directory

    for key in session_clusters.keys():
        filename = directory + str(key) + "_all_clusters.pkl" 
        save(filename, session_clusters[key]) 

# TODO, Issue #188
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

# TODO, Issue #159
def get_all_dropout(IDS,version=None,hit_threshold=0,verbose=False): 
    '''
        For each session in IDS, returns the vector of dropout scores for each model
    '''

    directory=pgt.get_directory(version,subdirectory='summary')

    all_dropouts = []
    hits = []
    false_alarms = []
    correct_reject = []
    misses = []
    bsids = []
    crashed = 0
    low_hits = 0
    
    # Loop through IDS, add information from sessions above hit threshold
    for bsid in tqdm(IDS):
        try:
            fit = load_fit(bsid,version=version)
            if np.sum(fit['psydata']['hits']) >= hit_threshold:
                dropout_dict = get_session_dropout(fit)
                dropout = [dropout_dict[x] for x in sorted(list(fit['weights'].keys()))] 
                all_dropouts.append(dropout)
                hits.append(np.sum(fit['psydata']['hits']))
                false_alarms.append(np.sum(fit['psydata']['false_alarms']))
                correct_reject.append(np.sum(fit['psydata']['correct_reject']))
                misses.append(np.sum(fit['psydata']['misses']))
                bsids.append(bsid)
            else:
                low_hits+=1
        except:
            if verbose:
                print(str(bsid) +" crash")
            crashed +=1

    print(str(crashed) + " crashed")
    print(str(low_hits) + " below hit threshold")
    dropouts = np.stack(all_dropouts,axis=1)
    filepath = directory + "all_dropouts.pkl"
    save(filepath, dropouts)
    return dropouts,hits, false_alarms, misses,bsids, correct_reject

# TODO, Issue #159
def load_all_dropout(version=None):
    directory = pgt.get_directory(version,subdirectory='summary')
    dropout = load(directory+"all_dropouts.pkl")
    return dropout

# TODO, Issue #187
def get_mice_weights(mice_ids,version=None,hit_threshold=0,verbose=False,manifest = None):
    directory=pgt.get_directory(version)
    if manifest is None:
        manifest = pgt.get_ophys_manifest()
    mice_weights = []
    mice_good_ids = []
    crashed = 0
    low_hits = 0
    # Loop through IDS
    for id in tqdm(mice_ids):
        this_mouse = []
        for sess in manifest.query('donor_id == @id').behavior_session_id.values:
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

# TODO, Issue #187
def get_mice_dropout(mice_ids,version=None,hit_threshold=0,verbose=False,manifest=None):

    directory=pgt.get_directory(version)    
    if manifest is None:
        manifest = pgt.get_ophys_manifest()

    mice_dropouts = []
    mice_good_ids = []
    crashed = 0
    low_hits = 0

    # Loop through IDS
    for id in tqdm(mice_ids):
        this_mouse = []
        for sess in manifest.query('donor_id ==@id')['behavior_session_id'].values:
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

# TODO, Issue #190
def PCA_dropout(ids,mice_ids,version,verbose=False,hit_threshold=0,manifest=None,ms=2):
    dropouts, hits,false_alarms,misses,ids,correct_reject = get_all_dropout(ids,
        version,verbose=verbose,hit_threshold=hit_threshold)

    mice_dropouts, mice_good_ids = get_mice_dropout(mice_ids,
        version=version,verbose=verbose,hit_threshold=hit_threshold,
        manifest = manifest)

    fit = load_fit(ids[1],version=version)
    labels = sorted(list(fit['weights'].keys()))
    pca,dropout_dex,varexpl = PCA_on_dropout(dropouts, labels=labels,
        mice_dropouts=mice_dropouts,mice_ids=mice_good_ids, hits=hits,
        false_alarms=false_alarms, misses=misses,version=version, correct_reject = correct_reject,ms=ms)

    return dropout_dex,varexpl

# TODO, Issue #190
def PCA_on_dropout(dropouts,labels=None,mice_dropouts=None, mice_ids = None,hits=None,false_alarms=None, misses=None,version=None,fs1=12,fs2=12,filetype='.png',ms=2,correct_reject=None):
    directory=pgt.get_directory(version)
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

# TODO, Issue #190
def PCA_weights(ids,mice_ids,version=None,verbose=False,manifest = None,hit_threshold=0):
    directory=pgt.get_directory(version)
    #all_weights,good_ids =plot_session_summary_weights(ids,return_weights=True,version=version,hit_threshold=hit_threshold)
    plot_session_summary_weights(ids,return_weights=True,version=version,hit_threshold=hit_threshold)
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
    labels = pgt.get_clean_string(weights_list)    
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

# TODO, Issue #190
def PCA_analysis(ids, mice_ids,version,hit_threshold=0,manifest=None):
    # PCA on dropouts
    drop_dex,drop_varexpl = PCA_dropout(ids,mice_ids,version,hit_threshold=hit_threshold,manifest=manifest)

    # PCA on weights
    weight_dex,weight_varexpl = PCA_weights(ids,mice_ids,version,manifest=manifest)
   
    # Compare
    directory=pgt.get_directory(version) 
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

   
# TODO, Issue #187
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

# TODO, Issue #187
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
        
# TODO, Issue #187
def mouse_roc(fit):
    fit['roc_session'] = []
    for i in range(0,len(fit['psydata']['dayLength'])):
        data = copy.copy(fit['psydata_session'][i]-1)
        model = copy.copy(fit['cv_pred_session'][i])
        fit['roc_session'].append(metrics.roc_auc_score(data,model))

# TODO, Issue #187
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

# TODO, Issue #187
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

# TODO, Issue #187
def plot_all_mouse_session_roc(directory):
    rocs = load(directory+"all_roc_session_mouse.pkl")
    plt.figure()
    plt.plot(np.array(rocs[1])*100, np.array(rocs[0])*100,'ko')
    plt.plot([60,100],[60,100],'k--')
    plt.xlabel('Session ROC (%)')
    plt.ylabel('Mouse ROC (%)')
    plt.savefig(directory+"all_roc_session_mouse.png") 

# TODO, Issue #187
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

# TODO, Issue #187
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


# TODO, Issue #201
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
   
 
# TODO, Issue #173
def get_timing_index_fit(fit,return_all=False):
    '''
        
    '''
    dropout = get_session_dropout(fit)
    model_dex = -(dropout['task0'] - dropout['timing1D'])
    if return_all:
        return model_dex, dropout['task0'], dropout['timing1D']
    else:
        return model_dex   


# TODO, Issue #173
def get_cross_validation_dropout(cv_results):
    '''
        computes the full log likelihood by summing each cross validation fold
    '''
    return np.sum([i['logli'] for i in cv_results]) 


 # TODO, Issue #173
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


def plot_task_timing_by_training_duration(model_manifest,version=None, savefig=True,group_label='all'):
    
    raise Exception('Need to update')
    #TODO Issue, #92
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
        directory=pgt.get_directory(version)
        plt.savefig(directory+'figures_summary/'+group_label+"_task_index_by_train_duration.png")


