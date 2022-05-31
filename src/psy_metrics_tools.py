import numpy as np
import psy_general_tools as pgt
from scipy.stats import norm

'''
This is a set of functions for calculating and analyzing model free behavioral metrics on a flash by flash basis
Alex Piet, alexpiet@gmail.com
11/5/2019

'''

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

    # For numerical stability, I wipe results and re-annotate 
    if 'bout_number' in session.licks:
        session.licks.drop(columns=['pre_ili','post_ili','rewarded',
            'bout_start','bout_end','bout_number','bout_rewarded','bout_num_rewards','num_rewards'],
            inplace=True,errors='ignore')

    # Remove licks that happen outside of stimulus period
    stim_start = session.stimulus_presentations.query('not omitted')['start_time'].values[0]
    stim_end   = session.stimulus_presentations['start_time'].values[-1]+0.75
    session.licks.query('(timestamps >= @stim_start) and (timestamps <= @stim_end)',
        inplace=True)
    session.licks.reset_index(drop=True,inplace=True)

    # Computing ILI for each lick 
    session.licks['pre_ili'] = np.concatenate([
        [np.nan],np.diff(session.licks.timestamps.values)])
    session.licks['post_ili'] = np.concatenate([
        np.diff(session.licks.timestamps.values),[np.nan]])

    # Segment licking bouts
    session.licks['bout_start'] = session.licks['pre_ili'] > bout_threshold
    session.licks['bout_end'] = session.licks['post_ili'] > bout_threshold
    session.licks.at[session.licks['pre_ili'].apply(np.isnan),'bout_start']=True
    session.licks.at[session.licks['post_ili'].apply(np.isnan),'bout_end']=True
    session.licks['bout_number'] = np.cumsum(session.licks['bout_start'])

    # Annotate rewards
    # Iterate through rewards
    session.licks['rewarded'] = False # Setting default to False
    session.licks['num_rewards'] = 0 
    for index, row in session.rewards.iterrows():
        if row.autorewarded:
            # Assign to nearest lick
            mylick = np.abs(session.licks.timestamps - row.timestamps).idxmin()
        else:
            # Assign reward to last lick before reward time
            this_reward_lick_times = np.where(session.licks.timestamps <= row.timestamps)[0]
            if len(this_reward_lick_times) == 0:
                raise Exception('First lick was after first reward')
            else:
                mylick = this_reward_lick_times[-1]
        session.licks.at[mylick,'rewarded'] = True 
        # licks can be double assigned to rewards because of auto-rewards
        session.licks.at[mylick,'num_rewards'] +=1  

    # Annotate bout rewards  
    x = session.licks.groupby('bout_number').any('rewarded').rename(columns={'rewarded':'bout_rewarded'})
    y = session.licks.groupby('bout_number')['num_rewards'].sum().rename(columns={'num_rewards':'bout_num_rewards'})
    session.licks['bout_rewarded'] = False
    temp = session.licks.reset_index().set_index('bout_number')
    temp.update(x)
    temp['bout_num_rewards'] = y
    temp = temp.reset_index().set_index('index')
    session.licks['bout_rewarded'] = temp['bout_rewarded']
    session.licks['bout_num_rewards'] = temp['bout_num_rewards']

    # QC
    # Check that all rewards are matched to a lick
    num_lick_rewards = session.licks['rewarded'].sum()
    num_rewards = len(session.rewards)
    double_rewards = np.sum(session.licks.query('num_rewards >1')['num_rewards']-1)
    #if num_rewards != num_lick_rewards+double_rewards: # TODO DEBUG
    #    print('num rewards ({}) dont match num_lick_rewards ({})'.format(num_rewards, num_lick_rewards))
    assert num_rewards == num_lick_rewards+double_rewards, \
        "Lick Annotations don't match number of rewards"

    # Check that all rewards are matched to a bout
    num_rewarded_bouts=np.sum(session.licks['bout_rewarded']&session.licks['bout_start'])
    double_rewarded_bouts = np.sum(session.licks[session.licks['bout_rewarded']&session.licks['bout_start']&(session.licks['bout_num_rewards']>1)]['bout_num_rewards']-1)
    #if num_rewards != num_rewarded_bouts: # TODO DEBUG
    #    print('rewarded bout annotations ({}) dont match number of rewards ({})'.format(num_rewarded_bouts, num_rewards))
    #    print(double_rewarded_bouts)
    assert num_rewards == num_rewarded_bouts+double_rewarded_bouts, \
        "Bout Annotations don't match number of rewards"
 
    # Check that bouts start and stop
    num_bout_start = session.licks['bout_start'].sum()
    num_bout_end = session.licks['bout_end'].sum()
    num_bouts = session.licks['bout_number'].max()
    assert num_bout_start==num_bout_end, "Bout Starts and Bout Ends don't align"
    assert num_bout_start == num_bouts, "Number of bouts is incorrect"

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
            session.stimulus_presentations.at[session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)].index[0]-1,'bout_number'] = x.bout_number
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

    # Add engagement classification
    reward_threshold = pgt.get_engagement_threshold()
    session.stimulus_presentations['engaged'] = [x > reward_threshold for x in session.stimulus_presentations['reward_rate']]


