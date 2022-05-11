import numpy as np
import psy_style as pstyle
import matplotlib.pyplot as plt
import psy_general_tools as pgt
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
import scipy.stats as ss
from scipy.stats import norm
from scipy import stats
from tqdm import tqdm

'''
This is a set of functions for calculating and analyzing model free behavioral metrics on a flash by flash basis
Alex Piet, alexpiet@gmail.com
11/5/2019

'''
MODEL_FREE_DIR = '/home/alex.piet/codebase/behavior/model_free/'
# TODO, Issue #176
def get_metrics(session,add_running=False):
    '''
        Top level function that appends a few columns to session.stimulus_presentations,
            and a few columns to session.licks 

        ARGUMENTS: session, SDK session object
        
        Adds to session.licks
            pre_ili,        (seconds)
            post_ili,       (seconds)
            rewarded,       (boolean)
            bout_start,     (boolean)
            bout_end,       (boolean)
            bout_number,    (int)
            bout_rewarded,  (boolean)

        Adds to session.stimulus_presentations
            bout_start,     (boolean)
            bout_end,       (boolean)
            licked,         (boolean)
            lick_rate,      (licks/flash)
            rewarded,       (boolean)
            reward_rate,    (rewards/flash)
            running_rate,
            bout_rate,      (bouts/flash)
            high_lick,      (boolean)
            high_reward,    (boolean)
            flash_metrics_epochs, (int)
            flash_metrics_labels, (string)
    '''
    annotate_licks(session)
    annotate_bouts(session)
    annotate_flash_rolling_metrics(session,add_running=add_running)
    classify_by_flash_metrics(session)

# TODO, Issue #176
def annotate_licks(session,bout_threshold=0.7):
    '''
        Appends several columns to session.licks. Calculates licking bouts based on a
        interlick interval (ILI) of bout_threshold. Default of 700ms based on examining 
        histograms of ILI distributions

        Adds to session.licks
            pre_ili,        (seconds)
            post_ili,       (seconds)
            rewarded,       (boolean)
            bout_start,     (boolean)
            bout_end,       (boolean)
            bout_number,    (int)
            bout_rewarded,  (boolean)
    '''

    # Something was buggy upon repeated re-annotations, so I throw an error
    if 'bout_number' in session.licks:
        raise Exception('You already annotated this session, reload session first')

    # Computing ILI for each lick 
    licks = session.licks
    licks['pre_ili'] = np.concatenate([[np.nan],np.diff(licks.timestamps.values)])
    licks['post_ili'] = np.concatenate([np.diff(licks.timestamps.values),[np.nan]])
    licks['rewarded'] = False
    for index, row in session.rewards.iterrows():
        if len(np.where(licks.timestamps<=row.timestamps)[0]) == 0:
            if (row.autorewarded) & (row.timestamps <= licks.timestamps.values[0]):
                # mouse hadn't licked before first auto-reward
                mylick = 0
            else:
                print('First lick was after first reward, but it wasnt an auto-reward. This is very strange, but Im annotating the first lick as rewarded.')
                mylick = 0
        else:
            mylick = np.where(licks.timestamps <= row.timestamps)[0][-1]
        licks.at[mylick,'rewarded'] = True
    
    # Segment licking bouts
    licks['bout_start'] = licks['pre_ili'] > bout_threshold
    licks['bout_end'] = licks['post_ili'] > bout_threshold
    licks.at[licks['pre_ili'].apply(np.isnan),'bout_start']=True
    licks.at[licks['post_ili'].apply(np.isnan),'bout_end']=True

    # Annotate bouts by number, and reward
    licks['bout_number'] = np.cumsum(licks['bout_start'])
    x = session.licks.groupby('bout_number').any('rewarded').rename(columns={'rewarded':'bout_rewarded'})
    session.licks['bout_rewarded'] = False
    temp = session.licks.reset_index().set_index('bout_number')
    temp.update(x)
    temp = temp.reset_index().set_index('index')
    session.licks['bout_rewarded'] = temp['bout_rewarded']

# TODO, Issue #176
def annotate_bouts(session):
    '''
        Uses the bout annotations in session.licks to annotate session.stimulus_presentations

        Adds to session.stimulus_presentations
            bout_start,     (boolean)
            bout_end,       (boolean)

    '''
    # Annotate Bout Starts
    bout_starts = session.licks[session.licks['bout_start']]
    session.stimulus_presentations['bout_start'] = False
    session.stimulus_presentations['num_bout_start'] = 0
    for index,x in bout_starts.iterrows():
        filter_start = session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)]
        if (x.timestamps > session.stimulus_presentations.iloc[0].start_time ) & (len(filter_start) > 0):
            session.stimulus_presentations.at[session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)].index[0]-1,'bout_start'] = True
            session.stimulus_presentations.at[session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)].index[0]-1,'num_bout_start'] += 1
    # Annotate Bout Ends
    bout_ends = session.licks[session.licks['bout_end']]
    session.stimulus_presentations['bout_end'] = False
    session.stimulus_presentations['num_bout_end'] = 0
    for index,x in bout_ends.iterrows():
        filter_start = session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)]
        if (x.timestamps > session.stimulus_presentations.iloc[0].start_time) & (len(filter_start) > 0):
            session.stimulus_presentations.at[session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)].index[0]-1,'bout_end'] = True
            session.stimulus_presentations.at[session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)].index[0]-1,'num_bout_end'] += 1
            # Check to see if bout started before stimulus, if so, make first flash as bout_starts
            bout_start_time = session.licks.query('bout_number == @x.bout_number').query('bout_start').timestamps.values[0]
            bout_end_time = x.timestamps
            if (bout_start_time < session.stimulus_presentations.iloc[0].start_time) & (bout_end_time > session.stimulus_presentations.iloc[0].start_time):
                session.stimulus_presentations.at[0,'bout_start'] = True
                session.stimulus_presentations.at[0,'num_bout_start'] += 1
    # Clean Up
    session.stimulus_presentations.drop(-1,inplace=True,errors='ignore')

# TODO, Issue #176
def annotate_bout_start_time(session):
    session.stimulus_presentations['bout_start_time'] = np.nan
    session.stimulus_presentations.at[session.stimulus_presentations['bout_start'] == True,'bout_start_time'] = session.stimulus_presentations[session.stimulus_presentations['bout_start']==True].licks.str[0]
    

# TODO, Issue #176
def annotate_flash_rolling_metrics(session,win_dur=320, win_type='triang', add_running=False):
    '''
        Get rolling flash level metrics for lick rate, reward rate, and bout_rate
        Computes over a rolling window of win_dur (s) duration, with a window type given by win_type

        Adds to session.stimulus_presentations
            licked,         (boolean)
            lick_rate,      (licks/flash)
            rewarded,       (boolean)
            reward_rate,    (rewards/flash)
            running_rate,   (cm/s)
            bout_rate,      (bouts/flash)
    '''
    # Get Lick Rate / second
    if 'licked' not in session.stimulus_presentations:
        session.stimulus_presentations['licked'] = [len(this_lick) > 0 for this_lick in session.stimulus_presentations['licks']]
    session.stimulus_presentations['lick_rate'] = session.stimulus_presentations['licked'].rolling(win_dur, min_periods=1,win_type=win_type).mean()/.75

    # Get Reward Rate / second
    session.stimulus_presentations['rewarded'] = [len(this_reward) > 0 for this_reward in session.stimulus_presentations['rewards']]
    session.stimulus_presentations['reward_rate'] = session.stimulus_presentations['rewarded'].rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75

    # Get Running / Second
    if add_running:
        session.stimulus_presentations['running_rate'] = session.stimulus_presentations['mean_running_speed'].rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75

    # Get Bout Rate / second
    session.stimulus_presentations['bout_rate'] = session.stimulus_presentations['bout_start'].rolling(win_dur,min_periods=1, win_type=win_type).mean()/.75

    # Get Hit Fraction. % of licks that are rewarded
    session.stimulus_presentations['hit_bout'] = [
        np.nan if (not x[0]) else 1 if (x[1]==1) else 0 
        for x in zip(session.stimulus_presentations['bout_start'], session.stimulus_presentations['rewarded'])]
    session.stimulus_presentations['lick_hit_fraction'] = \
        session.stimulus_presentations['hit_bout'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)
    
    # Get Hit Rate, % of change flashes with licks
    session.stimulus_presentations['change_with_lick'] = [
        np.nan if (not x[0]) else 1 if (x[1]) else 0 
        for x in zip(session.stimulus_presentations['is_change'],session.stimulus_presentations['bout_start'])]
    session.stimulus_presentations['hit_rate'] = \
        session.stimulus_presentations['change_with_lick'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)
  
    # Get Miss Rate, % of change flashes without licks
    session.stimulus_presentations['change_without_lick'] = [
        np.nan if (not x[0]) else 0 if (x[1]) else 1 
        for x in zip(session.stimulus_presentations['is_change'],session.stimulus_presentations['bout_start'])]
    session.stimulus_presentations['miss_rate'] = \
        session.stimulus_presentations['change_without_lick'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)

    # Get False Alarm Rate, % of non-change flashes with licks
    session.stimulus_presentations['non_change_with_lick'] = [
        np.nan if (x[0]) else 1 if (x[1]) else 0 
        for x in zip(session.stimulus_presentations['is_change'],session.stimulus_presentations['bout_start'])]
    session.stimulus_presentations['false_alarm_rate'] = \
        session.stimulus_presentations['non_change_with_lick'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)

    # Get Correct Reject Rate, % of non-change flashes without licks
    session.stimulus_presentations['non_change_without_lick'] = [
        np.nan if (x[0]) else 0 if (x[1]) else 1 
        for x in zip(session.stimulus_presentations['is_change'],session.stimulus_presentations['bout_start'])]
    session.stimulus_presentations['correct_reject_rate'] = \
        session.stimulus_presentations['non_change_without_lick'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)

    # Get dPrime and Criterion metrics on a flash level
    Z = norm.ppf
    session.stimulus_presentations['d_prime']   = Z(np.clip(session.stimulus_presentations['hit_rate'],0.01,0.99)) - Z(np.clip(session.stimulus_presentations['false_alarm_rate'],0.01,0.99)) 
    session.stimulus_presentations['criterion'] = 0.5*(Z(np.clip(session.stimulus_presentations['hit_rate'],0.01,0.99)) + Z(np.clip(session.stimulus_presentations['false_alarm_rate'],0.01,0.99)))
        # Computing the criterion to be negative
    
    # Add Reaction Time
    session.stimulus_presentations['RT'] = [x[0][0]-x[1] if (len(x[0]) > 0) &x[2] else np.nan for x in zip(session.stimulus_presentations['licks'], session.stimulus_presentations['start_time'], session.stimulus_presentations['bout_start'])]

 # TODO, Issue #176 #TODO, Issue #213
def classify_by_flash_metrics(session, lick_threshold = 0.1, reward_threshold=1/90,use_bouts=True):
    '''
        Use the flash level rolling metrics to classify into three states based on the thresholds
        lick_threshold is the licking rate / flash that divides high and low licking states
        reward_threshold is the rewards/flash that divides high and low reward states (2/80 is equivalent to 2 rewards/minute). 
        OLD: 0.1, lick, 2/80 reward
    '''
    #if use_bouts:
    #    session.stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in session.stimulus_presentations['bout_rate']] 
    #else:
    #    session.stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in session.stimulus_presentations['lick_rate']] 
    #session.stimulus_presentations['high_reward'] = [True if x > reward_threshold else False for x in session.stimulus_presentations['reward_rate']] 
    #session.stimulus_presentations['flash_metrics_epochs'] = [0 if (not x[0]) & (not x[1]) else 1 if x[1] else 2 for x in zip(session.stimulus_presentations['high_lick'], session.stimulus_presentations['high_reward'])]
    #session.stimulus_presentations['flash_metrics_labels'] = ['low-lick,low-reward' if x==0  else 'high-lick,high-reward' if x==1 else 'high-lick,low-reward' for x in session.stimulus_presentations['flash_metrics_epochs']]
    session.stimulus_presentations['engaged'] = [x > reward_threshold for x in session.stimulus_presentations['reward_rate']]

# TODO, Issue #176 #TODO, Issue #213
def get_engagement_for_fit(fit, lick_threshold=0.1, reward_threshold=1/90, use_bouts=True,win_dur=320, win_type='triang'):
    fit['psydata']['full_df']['bout_rate'] = fit['psydata']['full_df']['bout_start'].rolling(win_dur,min_periods=1, win_type=win_type).mean()/.75
    #fit['psydata']['full_df']['high_lick'] = [True if x > lick_threshold else False for x in fit['psydata']['full_df']['bout_rate']] 
    fit['psydata']['full_df']['reward_rate'] = fit['psydata']['full_df']['hits'].rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75
    #fit['psydata']['full_df']['high_reward'] = [True if x > reward_threshold else False for x in fit['psydata']['full_df']['reward_rate']] 
    #fit['psydata']['full_df']['flash_metrics_epochs'] = [0 if (not x[0]) & (not x[1]) else 1 if x[1] else 2 for x in zip(fit['psydata']['full_df']['high_lick'], fit['psydata']['full_df']['high_reward'])]
    #fit['psydata']['full_df']['flash_metrics_labels'] = ['low-lick,low-reward' if x==0  else 'high-lick,high-reward' if x==1 else 'high-lick,low-reward' for x in fit['psydata']['full_df']['flash_metrics_epochs']]
    #fit['psydata']['full_df']['engaged'] = [(x=='high-lick,low-reward') or (x=='high-lick,high-reward') for x in fit['psydata']['full_df']['flash_metrics_labels']]
    fit['psydata']['full_df']['engaged'] = [x > reward_threshold for x in fit['psydata']['full_df']['reward_rate']]
    return fit


