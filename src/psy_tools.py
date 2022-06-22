import os
import copy
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

import psytrack as psy
from psytrack.helper.crossValidation import split_data
from psytrack.helper.crossValidation import xval_loglike

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
    plot_weights(wMode, weights,psydata,errorbar=credibleInt, ypred=ypred,
        filename=fig_filename)

    print("Cross Validation Analysis")
    cross_psydata = psy.trim(psydata, 
        END=int(np.floor(len(psydata['y'])/format_options['num_cv_folds'])\
        *format_options['num_cv_folds'])) 
    cross_results = compute_cross_validation(cross_psydata, hyp, weights,
        folds=format_options['num_cv_folds'])
    cv_pred = compute_cross_validation_ypred(cross_psydata, cross_results,ypred)
    
    if complete:
        print("Dropout Analysis")
        models = dropout_analysis(psydata, strategies, format_options)

    print('Packing up and saving')
    metadata = session.metadata
    output = [ hyp,   evd,   wMode,   hess,   credibleInt,   weights,   ypred,  
        psydata,  cross_results,  cv_pred,  metadata]
    labels = ['hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred',
        'psydata','cross_results','cv_pred','metadata']       
    fit = dict((x,y) for x,y in zip(labels, output))
    fit['ID'] = bsid

    if complete:
        fit['models'] = models

    print('Saving fit dictionary')
    save(filename+".pkl", fit) 
    summarize_fit(fit, version=20, savefig=True)
    plt.close('all')

    print('Saving strategy df')
    build_session_strategy_df(bsid, version,fit=fit,session=session)

    print('Saving licks df')
    build_session_licks_df(session, bsid, version)

    print('Done!')


def build_session_strategy_df(bsid, version,TRAIN=False,fit=None,session=None):
    '''
        Saves an analysis file in <output_dir> for the model fit of session <id> 
        Extends model weights to be constant during licking bouts

        licked              (bool)  Did the mouse lick during this image?
        lick_bout_start     (bool)  did the mouse start a lick bout during this image?
        bout_number         (int)   oridinal count of licking bouts, only defined
                                    at the start of licking bouts
        lick_bout_end       (bool)  did a lick bout end during this image?
        in_lick_bout        (bool)   Whether this was an image removed for fitting
                                    because the animal was in a licking bout
        num_licks           (int)   Number of licks during this image 
        lick_rate           (float) licks/second 
        rewarded            (bool)  Whether this image was rewarded
        reward_rate         (float) rewards/second
        lick_bout_rate      (float) lick-bouts/second
        lick_hit_fraction   (float) % Percentage of lick bouts that were rewarded 
        hit_rate            (float) % Percentage of changes with rewards
        miss_rate           (float) % Percentage of changes without rewards
        false_alarm_rate    (float) % Percentage of non-changes with licks
        correct_reject_rate (float) % Percentage of non-changes without licks
        d_prime             (float)
        criterion           (float)
        RT                  (float) Response time from image onset in s
        engaged             (boolean)
        strategy weights            model weight for this image
            bias            (float)
            omissions       (float)
            omissions1      (float)
            task0           (float)
            timing1D        (float) 
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
            pm.annotate_image_rolling_metrics(session)

    # Load Model fit
    if fit is None:
        fit = load_fit(bsid, version=version)
 
    # include model weights
    weights = get_weights_list(fit['weights'])
    for wdex, weight in enumerate(weights):
        # Weights are not defined during lick bouts
        session.stimulus_presentations.at[\
            ~session.stimulus_presentations['in_lick_bout'], weight] = \
            fit['wMode'][wdex,:]

        # Fill in lick bouts with the value from the start of the bout
        session.stimulus_presentations[weight] = \
            session.stimulus_presentations[weight].fillna(method='ffill')
    
    # Clean up Stimulus Presentations
    model_output = session.stimulus_presentations.copy()
        
    # Drop columns from pm.annotations, and stimulus_presentations
    model_output.drop(columns=['duration', 'end_frame', 'image_set','index', 
        'orientation', 'start_frame', 'start_time', 'stop_time', 'licks', 
        'rewards', 'time_from_last_lick', 'time_from_last_reward', 
        'time_from_last_change', 'mean_running_speed', 'num_bout_start', 
        'num_bout_end','hit_bout'],inplace=True,errors='ignore') 

    # Drop columns that come from ps.annotate_stimulus_presentations
    model_output.drop(columns=['in_grace_period','correct_reject','misses',
        'auto_rewards','hits','aborts','false_alarm'],inplace=True, errors='ignore')

    # Clean up some names created in psy_metrics
    model_output = model_output.rename(columns={
        'bout_end':'lick_bout_end', 
        'bout_start':'lick_bout_start',
        'bout_rate':'lick_bout_rate',
        'change_with_lick':'hit',
        'change_without_lick':'miss',
        'non_change_with_lick':'image_false_alarm',
        'non_change_without_lick':'image_correct_reject'
        })

    # Save out dataframe
    model_output.to_csv(pgt.get_directory(version, \
        subdirectory='strategy_df')+str(bsid)+'.csv') 


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
        Adds columns to the stimulus_presentation table describing whether 
        certain task events happened during that image. Importantly! These annotations
        are just used for visualization purposes. 

        Inputs:
        session, the SDK session object
    
        Appends columns:
        hits,   True if the mouse licked on a change image. 
        misses, True if the mouse did not lick on a change image
        aborts, True if the mouse licked on a non-change-image. 
            THIS IS NOT THE SAME AS THE TRIALS TABLE ABORT DEFINITION.
            licks on sequential imagees that are during the abort time 
            out period are counted as aborts here.
            this abort list should only be used for simple visualization purposes
        in_grace_period, True if this image occurs during the 0.75 - 4.5 period 
            after the onset of a hit change
        false_alarm,    True if the mouse licked on a sham-change-image
        correct_reject, True if the mouse did not lick on a sham-change-image
        auto_rewards,   True if there was an auto-reward during this image
    '''
    session.stimulus_presentations['hits']   =  \
        session.stimulus_presentations['licked'] & \
        session.stimulus_presentations['is_change']
    session.stimulus_presentations['misses'] = \
        ~session.stimulus_presentations['licked'] & \
        session.stimulus_presentations['is_change']
    session.stimulus_presentations['aborts'] =  \
        session.stimulus_presentations['licked'] & \
        ~session.stimulus_presentations['is_change']
    session.stimulus_presentations['in_grace_period'] = \
        (session.stimulus_presentations['time_from_last_change'] <= 4.5) & \
        (session.stimulus_presentations['time_from_last_reward'] <=4.5)
    # Remove Aborts that happened during grace period
    session.stimulus_presentations.at[\
        session.stimulus_presentations['in_grace_period'],'aborts'] = False 

    # These ones require iterating the trials table, and is super slow
    session.stimulus_presentations['false_alarm'] = False
    session.stimulus_presentations['correct_reject'] = False
    session.stimulus_presentations['auto_rewards'] = False
    try:
        for i in session.stimulus_presentations.index:
            found_it=True
            trial = session.trials[
                (session.trials.start_time <= \
                session.stimulus_presentations.at[i,'start_time']) & 
                (session.trials.stop_time >=\
                session.stimulus_presentations.at[i,'start_time'] + 0.25)
                ]
            if len(trial) > 1:
                raise Exception("Could not isolate a trial for this image")
            if len(trial) == 0:
                trial = session.trials[
                    (session.trials.start_time <= \
                    session.stimulus_presentations.at[i,'start_time']) & 
                    (session.trials.stop_time+0.75 >= \
                    session.stimulus_presentations.at[i,'start_time'] + 0.25)
                    ]  
                if ( len(trial) == 0 ) & \
                    (session.stimulus_presentations.at[i,'start_time'] > \
                    session.trials.start_time.values[-1]):
                    trial = session.trials[\
                        session.trials.index == session.trials.index[-1]]
                elif ( len(trial) ==0) & \
                    (session.stimulus_presentations.at[i,'start_time'] < \
                    session.trials.start_time.values[0]):
                    trial = session.trials[session.trials.index == \
                    session.trials.index[0]]
                elif np.sum(session.trials.aborted) == 0:
                    found_it=False
                elif len(trial) == 0:
                    trial = session.trials[
                        (session.trials.start_time <= \
                        session.stimulus_presentations.at[i,'start_time']+0.75) & 
                        (session.trials.stop_time+0.75 >= \
                        session.stimulus_presentations.at[i,'start_time'] + 0.25)
                        ]  
                    if len(trial) == 0: 
                        print('stim index: '+str(i))
                        raise Exception("Could not find a trial for this image")
            if found_it:
                if trial['false_alarm'].values[0]:
                    if (trial.change_time.values[0] >= \
                        session.stimulus_presentations.at[i,'start_time']) & \
                        (trial.change_time.values[0] <= \
                        session.stimulus_presentations.at[i,'stop_time'] ):
                        session.stimulus_presentations.at[i,'false_alarm'] = True
                if trial['correct_reject'].values[0]:
                    if (trial.change_time.values[0] >= \
                        session.stimulus_presentations.at[i,'start_time']) & \
                        (trial.change_time.values[0] <= \
                        session.stimulus_presentations.at[i,'stop_time'] ):
                        session.stimulus_presentations.at[i,'correct_reject'] = True
                if trial['auto_rewarded'].values[0]:
                    if (trial.change_time.values[0] >= \
                        session.stimulus_presentations.at[i,'start_time']) & \
                        (trial.change_time.values[0] <= \
                        session.stimulus_presentations.at[i,'stop_time'] ):
                        session.stimulus_presentations.at[i,'auto_rewards'] = True
    except:
        if ignore_trial_errors:
            print('WARNING, had trial alignment errors, '+\
            'but are ignoring due to ignore_trial_errors=True')
        else:
            raise Exception('Trial Alignment Error. '+\
            'Set ignore_trial_errors=True to ignore. Image #: '+str(i))


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
                timing0/1, if True (Default), then timing is a vector of 0s and 1s, 
                    otherwise, is -1/+1
                mean_center, if True, then regressors are mean-centered
                timing_params, [p1,p2] parameters for 1D timing regressor
                timing_params_session, parameters custom fit for this session
                                
        Returns:
            data formated for psytrack. A dictionary with key/values:
            psydata['y'] = a vector of no-licks (1) and licks(2) for each images
            psydata['inputs'] = a dictionary with each key an input 
                ('random','timing', 'task', etc) each value has a 2D array of shape 
                (N,M), where N is number of imagees, and M is 1 unless you want to 
                look at history/image interaction terms
    '''     
    print('This session had {} licks and {} rewards'\
        .format(len(session.licks), len(session.rewards)))
    if len(session.licks) < 10:
        raise Exception('Less than 10 licks in this session')   

    # Build Dataframe of images
    annotate_stimulus_presentations(session,
        ignore_trial_errors = format_options['ignore_trial_errors'])
    columns = ['start_time','hits','misses','false_alarm','correct_reject',
        'aborts','auto_rewards','is_change','omitted','licked','bout_start',
        'bout_end','num_bout_start','num_bout_end','in_lick_bout']
    df = pd.DataFrame(data = session.stimulus_presentations[columns])
    df = df.rename(columns={'is_change':'change'})

    # Process behavior annotations
    df['y'] = np.array([2 if x else 1 for x in 
        session.stimulus_presentations.bout_start.values])
    df['images_since_last_lick'] = session.stimulus_presentations.groupby(\
        session.stimulus_presentations['bout_end'].cumsum()).cumcount(ascending=True)
    df['timing_input'] = [x+1 for x in df['images_since_last_lick'].shift(fill_value=0)]
    df['included'] = ~df['in_lick_bout']

    # Build Strategy regressors
    df['task0']      = np.array([1 if x else 0 for x in df['change']])
    df['task1']      = np.array([1 if x else -1 for x in df['change']])
    df['late_task0'] = df['task0'].shift(1,fill_value=0)
    df['late_task1'] = df['task1'].shift(1,fill_value=-1)
    df['taskCR']     = np.array([0 if x else -1 for x in df['change']])
    df['omissions']  = np.array([1 if x else 0 for x in df['omitted']])
    df['omissions1'] = np.array([x for x in np.concatenate([[0], 
                        df['omissions'].values[0:-1]])])

    # Build timing strategy using average timing parameters
    df['timing1D']          = np.array(\
        [timing_sigmoid(x,format_options['timing_params']) 
        for x in df['timing_input']])

    # Build timing strategy using session timing parameters
    df['timing1D_session']  = np.array(\
        [timing_sigmoid(x+1,format_options['timing_params_session']) 
        for x in df['images_since_last_lick'].shift(fill_value=0)])

    # Build 1-hot timing strategies
    if format_options['timing0/1']:
        min_timing_val = 0
    else:
        min_timing_val = -1
    df['timing1'] =  np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==0])
    df['timing2'] =  np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==1])
    df['timing3'] =  np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==2])
    df['timing4'] =  np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==3])
    df['timing5'] =  np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==4])
    df['timing6'] =  np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==5])
    df['timing7'] =  np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==6])
    df['timing8'] =  np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==7])
    df['timing9'] =  np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==8])
    df['timing10'] = np.array([1 if x else min_timing_val 
        for x in df['images_since_last_lick'].shift() ==9])

    # Segment out licking bouts
    full_df = copy.copy(df)
    df = df[df['included']] 
    df['missing_trials'] = np.concatenate([np.diff(df.index)-1,[0]])

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
                'image_ids': df.index.values,
                'df':df,
                'full_df':full_df }

    psydata['session_label'] = [session.metadata['session_type']]
    return psydata


def timing_sigmoid(x,params,min_val = -1, max_val = 0,tol=1e-3):
    '''
        Evaluates a sigmoid between min_val and max_val with parameters params
    '''
    if np.isnan(x):
        x = 0 
    y = min_val+(max_val-min_val)/(1+(x/params[1])**params[0])
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
            psydata['y'] = a vector of no-licks (1) and licks(2) for each images
            psydata['inputs'] = a dictionary with each key an input 
                ('random','timing', 'task', etc) each value has a 2D array of 
                shape (N,M), where N is number of imagees, and M is 1 unless 
                you want to look at history/image interaction terms

        RETURNS:
        hyp
        evd
        wMode
        hess
    '''
    # Set up number of regressors
    weights = {}
    for strat in strategies:
        weights[strat] = 1
    print(weights)
    K = np.sum([weights[i] for i in weights.keys()])

    # Set up initial hyperparameters
    hyper = {'sigInit': 2**4.,
            'sigma':[2**-4.]*K,
            'sigDay': 2**4}

    # Only used if we are fitting multiple sessions
    # where we have a different prior
    if fit_overnight:
        optList=['sigma','sigDay']
    else:
        optList=['sigma']
    
    # Do the fit
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


def plot_weights(wMode,weights,psydata,errorbar=None, ypred=None,START=0, END=0,
    plot_trials=True,session_labels=None, seedW = None,ypred_each = None,
    filename=None,smoothing_size=50):
    '''
        Plots the fit results by plotting the weights in linear and probability space. 
        wMode, the weights
        weights, the dictionary of strategyes
        psydata, the dataset
        errorbar, the std of the weights
        ypred, the full model prediction
        START, the image number to start on
        END, the image number to end on
     
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
        
    elif plot_trials:
        fig,ax = plt.subplots(nrows=3,ncols=1, figsize=(10,8))  
        
        trial_ax = 2
    elif (ypred is not None):
        fig,ax = plt.subplots(nrows=3,ncols=1, figsize=(10,8))
        
        full_ax = 2
    else:
        fig,ax = plt.subplots(nrows=2,ncols=1, figsize=(10,6))
        

    # Axis 0, plot weights
    for i in np.arange(0, len(weights_list)):
        ax[0].plot(wMode[i,:], linestyle="-", lw=3, color=my_colors[i],
            label=weights_list[i])        
        ax[0].fill_between(np.arange(len(wMode[i])), wMode[i,:]-2*errorbar[i], 
            wMode[i,:]+2*errorbar[i],facecolor=my_colors[i], alpha=0.1)    
        if seedW is not None:
            ax[0].plot(seedW[i,:], linestyle="--", lw=2, color=my_colors[i], 
                label= "seed "+weights_list[i])
    ax[0].plot([0,np.shape(wMode)[1]], [0,0], 'k--',alpha=0.2)
    ax[0].set_ylabel('Weight',fontsize=12)
    ax[0].set_xlabel('Image #',fontsize=12)
    ax[0].set_xlim(START,END)
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].tick_params(axis='both',labelsize=12)
    for i in np.arange(0, len(dayLength)-1):
        ax[0].axvline(dayLength[i],color='k',alpha=0.2)
        if session_labels is not None:
            ax[0].text(dayLength[i],ax[0].get_ylim()[1], session_labels[i],rotation=25)

    # Axis 1, plot nonlinear weights (approximation)
    for i in np.arange(0, len(weights_list)):
        ax[1].plot(transform(wMode[i,:]), linestyle="-", lw=3, color=my_colors[i],
            label=weights_list[i])
        ax[1].fill_between(np.arange(len(wMode[i])),transform(wMode[i,:]-2*errorbar[i]),
            transform(wMode[i,:]+2*errorbar[i]),facecolor=my_colors[i], alpha=0.1)                  
        if seedW is not None:
            ax[1].plot(transform(seedW[i,:]), linestyle="--", lw=2, color=my_colors[i],
                label= "seed "+weights_list[i])
    ax[1].set_ylim(0,1)
    ax[1].set_ylabel('Lick Prob',fontsize=12)
    ax[1].set_xlabel('Image #',fontsize=12)
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
        ax[2].set_yticklabels(['hits','miss','CR','FA','abort','auto'],
            fontdict={'fontsize':12})
        ax[2].set_xlim(START,END)
        ax[2].set_xlabel('Image #',fontsize=12)
        ax[2].tick_params(axis='both',labelsize=12)

    # Plot Full Model prediction and comparison with data
    if (ypred is not None):
        ax[full_ax].plot(pgt.moving_mean(ypred,smoothing_size), 'k',alpha=0.3,
            label='Full Model (n='+str(smoothing_size)+ ')')
        if ypred_each is not None:
            for i in np.arange(0, len(weights_list)):
                ax[full_ax].plot(ypred_each[:,i], linestyle="-", lw=3, 
                    alpha = 0.3,color=my_colors[i],label=weights_list[i])        
        ax[full_ax].plot(pgt.moving_mean(psydata['y']-1,smoothing_size), 'b',
            alpha=0.5,label='data (n='+str(smoothing_size)+ ')')
        ax[full_ax].set_ylim(0,1)
        ax[full_ax].set_ylabel('Lick Prob',fontsize=12)
        ax[full_ax].set_xlabel('Image #',fontsize=12)
        ax[full_ax].set_xlim(START,END)
        ax[full_ax].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax[full_ax].tick_params(axis='both',labelsize=12)
   
    # Save
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename+"_weights.png")
  
 
def compute_cross_validation(psydata, hyp, weights,folds=10):
    '''
        Computes Cross Validation for the data given the regressors as 
        defined in hyp and weights
    '''
    trainDs, testDs = split_data(psydata,F=folds)
    test_results = []
    for k in range(folds):
        print("\rrunning fold " +str(k),end="") 
        _,_,wMode_K,_ = psy.hyperOpt(trainDs[k], hyp, weights, ['sigma'],hess_calc=None)
        logli, gw = xval_loglike(testDs[k], wMode_K, trainDs[k]['missing_trials'], 
            weights)
        res = {'logli' : np.sum(logli), 'gw' : gw, 'test_inds' : testDs[k]['test_inds']}
        test_results += [res]
   
    print("") 
    return test_results


def compute_cross_validation_ypred(psydata,test_results,ypred):
    '''
        Computes the predicted outputs from cross validation results by stitching 
        together the predictions from each folds test set

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
        Computes a dropout analysis for the data in psydata. 
        In general, computes a full set, and then removes each feature one by one. 

        Returns a list of models and a list of labels for each dropout
    '''
    models =dict()

    hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,strategies)
    cross_psydata = psy.trim(psydata, 
        END=int(np.floor(len(psydata['y'])/format_options['num_cv_folds'])\
        *format_options['num_cv_folds'])) 
    cross_results = compute_cross_validation(cross_psydata, hyp, weights,
        folds=format_options['num_cv_folds'])
    models['Full'] = (hyp, evd, wMode, hess, credibleInt,weights,cross_results)

    # Iterate through strategies and remove them
    for s in strategies:
        dropout_strategies = copy.copy(strategies)
        dropout_strategies.remove(s)
        hyp, evd, wMode, hess, credibleInt,weights = fit_weights(psydata,
            dropout_strategies)
        cross_results = compute_cross_validation(cross_psydata, hyp, weights,
            folds=format_options['num_cv_folds'])
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
    '''
    directory = pgt.get_directory(version,subdirectory='fits')
    filename = directory + str(bsid) + ".pkl" 
    output = load(filename)
    if type(output) is not dict:
        labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', \
            'credibleInt', 'weights', 'ypred','psydata','cross_results',\
            'cv_pred','metadata']
        fit = dict((x,y) for x,y in zip(labels, output))
    else:
        fit = output
    fit['bsid'] = bsid
    return fit


def load_session_strategy_df(bsid, version, TRAIN=False):
    if TRAIN:
        raise Exception('need to implement')
    else:
        return pd.read_csv(pgt.get_directory(version, subdirectory='strategy_df')+\
            str(bsid)+'.csv') 


def load_session_licks_df(bsid, version):
    licks=pd.read_csv(pgt.get_directory(version,subdirectory='licks_df')+\
        str(bsid)+'.csv')
    licks['behavior_session_id'] = bsid
    return licks


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
        ax[0,1].plot(i,fit['hyp']['sigma'][i],'o',color=my_colors[i],\
            label=weights_list[i])
    ax[0,1].set_ylabel('Smoothing Prior, $\sigma$ \n <-- More Smooth'+\
        '      More Variable -->')
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

    fig.text(.7,starty-offset*1,"Driver Line:  " ,fontsize=fs,\
        horizontalalignment='right')
    fig.text(.7,starty-offset*1,fit['metadata']['driver_line'][-1],fontsize=fs)

    fig.text(.7,starty-offset*2,"Stage:  "     ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*2,str(fit['metadata']['session_type']),fontsize=fs)

    fig.text(.7,starty-offset*3,"ROC Train:  ",fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*3,str(round(roc_train,2)),fontsize=fs)

    fig.text(.7,starty-offset*4,"ROC CV:  "    ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*4,str(round(roc_cv,2)),fontsize=fs)

    fig.text(.7,starty-offset*5,"Lick Fraction:  ",fontsize=fs,\
        horizontalalignment='right')
    fig.text(.7,starty-offset*5,\
        str(round(fit['psydata']['full_df']['bout_start'].mean(),3)),fontsize=fs)

    fig.text(.7,starty-offset*6,"Lick Hit Fraction:  ",fontsize=fs,\
        horizontalalignment='right')
    lick_hit_fraction = fit['psydata']['full_df']['hits'].sum()\
        /fit['psydata']['full_df']['bout_start'].sum()
    fig.text(.7,starty-offset*6,str(round(lick_hit_fraction,3)),fontsize=fs)

    fig.text(.7,starty-offset*7,"Trial Hit Fraction:  ",fontsize=fs,\
        horizontalalignment='right')
    trial_hit_fraction = fit['psydata']['full_df']['hits'].sum()\
        /fit['psydata']['full_df']['change'].sum()
    fig.text(.7,starty-offset*7,str(round(trial_hit_fraction,3)),fontsize=fs)

    fig.text(.7,starty-offset*8,"Dropout Task/Timing Index:  " ,fontsize=fs,\
        horizontalalignment='right')
    fig.text(.7,starty-offset*8,str(round(get_timing_index_fit(fit)[0],2)),fontsize=fs) 

    fig.text(.7,starty-offset*9,"Weight Task/Timing Index:  " ,fontsize=fs,\
        horizontalalignment='right')
    fig.text(.7,starty-offset*9,str(round(get_weight_timing_index_fit(fit),2)),\
        fontsize=fs)  

    fig.text(.7,starty-offset*10,"Num Hits:  " ,fontsize=fs,horizontalalignment='right')
    fig.text(.7,starty-offset*10,np.sum(fit['psydata']['hits']),fontsize=fs)  

    plt.tight_layout()
    if savefig:
        filename = directory + 'figures_sessions/'+str(fit['ID'])+"_summary.png"
        plt.savefig(filename)
    

def plot_fit(ID, fit=None, version=None,savefig=False):
    '''
        Plots the fit associated with a session ID
        Needs the fit dictionary. If you pass these values into, 
            the function is much faster 
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
        plot_trials=True,filename=filename)

    summarize_fit(fit,version=version, savefig=savefig)

    return fit
  

def get_weight_timing_index_fit(fit):
    '''
        Return Task/Timing Index from average weights
        This is a slightly different definition that used in the summary table. 
        Since this is only used for single session plotting, I'm not going to bother
        updating it. See PR #263, and Issue #201 for details
    '''
    weights = get_weights_list(fit['weights'])
    wMode = fit['wMode']
    avg_weight_task   = np.mean(wMode[np.where(np.array(weights) == 'task0')[0][0],:])
    avg_weight_timing = np.mean(wMode[np.where(np.array(weights) == 'timing1D')[0][0],:])
    index = avg_weight_task - avg_weight_timing
    return index
   
 
def get_timing_index_fit(fit):
    '''
        Computes the strategy dropout index by taking the difference between the 
        task and visual strategies      
    '''
    dropout = get_session_dropout(fit)
    model_dex = -(dropout['task0'] - dropout['timing1D'])
    return model_dex, dropout['task0'], dropout['timing1D']


def get_cross_validation_dropout(cv_results):
    '''
        computes the full log likelihood by summing each cross validation fold
    '''
    return np.sum([i['logli'] for i in cv_results]) 


def get_session_dropout(fit, cross_validation=False):
    '''
        Compute the dropout scores for each strategy in this fit
        Can compute the dropout scores either using the cross-validated likelihood
        (cross_validation=True), or the model evidence (cross_validation=False)


        For each strategy fit['models'][<strategy>] is a tuple
        (hyp, evd, wMode, hess, credibleInt,weights,cross_results), 
        so we either compare evd or cross_results

        Returns a dictionary of strategies. 

    '''
    # Get list of strategies
    dropout = dict()
    models = sorted(list(fit['models'].keys()))
    models.remove('Full')
    
    # Iterate strategies and compute dropout scores
    if cross_validation:
         for m in models:
            this_CV = get_cross_validation_dropout(fit['models'][m][6])
            full_CV = get_cross_validation_dropout(fit['models']['Full'][6])
            dropout[m] = (1-this_CV/full_CV)*100   
    else:
        for m in models:
            this_ev = fit['models'][m][1]
            full_ev = fit['models']['Full'][1]
            dropout[m] = (1-this_ev/full_ev)*100
    
    return dropout   


def plot_task_timing_by_training_duration(model_manifest,version=None, savefig=True,
    group_label='all'):
    
    raise Exception('Need to update')
    #TODO Issue, #92
    avg_index = []
    num_train_sess = []
    behavior_sessions = pgt.get_training_manifest()
    behavior_sessions['training'] = behavior_sessions['ophys_session_id'].isnull()
    for index, mouse in enumerate(pgt.get_mice_ids()):
        df = behavior_sessions.query('donor_id ==@mouse')
        num_train_sess.append(len(df.query('training')))
        avg_index.append(model_manifest.query('donor_id==@mouse').\
            strategy_dropout_index.mean())

    plt.figure()
    plt.plot(avg_index, num_train_sess,'ko')
    plt.ylabel('Number of Training Sessions')
    plt.xlabel('Strategy Index')
    plt.axvline(0,ls='--',color='k')
    plt.axhline(0,ls='--',color='k')

    if savefig:
        directory=pgt.get_directory(version)
        plt.savefig(directory+'figures_summary/'+group_label+\
            "_task_index_by_train_duration.png")


