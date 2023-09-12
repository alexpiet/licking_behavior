import numpy as np
import psy_general_tools as pgt
from scipy.stats import norm

'''
This is a set of functions for calculating and analyzing model free behavioral 
metrics on a image by image basis
Alex Piet, alexpiet@gmail.com
11/5/2019

'''

def get_metrics(session):
    '''
        Top level function that appends a few columns to session.stimulus_presentations,
            and a few columns to session.licks 

        ARGUMENTS: session, SDK session object
        
        Adds to session.licks
            pre_ili,        (seconds)
            post_ili,       (seconds)
            bout_start,     (boolean) 
            bout_end,       (boolean) 
            bout_number,    (int)  
            rewarded,       (boolean) 
            num_rewards,    (int)  
            bout_rewarded,  (boolean) 
            bout_num_rewards,(int) 

        Adds to session.stimulus_presentations
            bout_start,     (boolean)
            num_bout_start, (int)
            bout_number,    (int)
            bout_end,       (boolean)
            num_bout_end,   (int)
            licked,         (boolean)
            lick_rate,      (licks/image)
            rewarded,       (boolean)
            reward_rate,    (rewards/image)
            bout_rate,      (bouts/image)
            high_lick,      (boolean)
            high_reward,    (boolean)
            image_metrics_epochs, (int)
            image_metrics_labels, (string)
    '''
    annotate_licks(session)
    annotate_bouts(session)
    annotate_image_rolling_metrics(session)
 

def annotate_licks(session):
    '''
        Appends several columns to session.licks. Calculates licking bouts based on a
        interlick interval (ILI) of bout_threshold. Default of 700ms based on examining 
        histograms of ILI distributions

        Adds to session.licks
            pre_ili,        (seconds)
            post_ili,       (seconds)
            bout_start,     (boolean) Whether this lick triggered a new licking bout
            bout_end,       (boolean) Whether this lick ended a licking bout
            bout_number,    (int) The label for the licking bout that contained
                            this lick
            rewarded,       (boolean) Whether this lick triggered a reward
            num_rewards,    (int) The number of rewards triggered to this lick. This is
                            only greater than 1 for auto-rewards, which are assigned
                            to the nearest lick. 
            bout_rewarded,  (boolean) Whether this bout triggered a reward.
            bout_num_rewards,(int) The number of rewards triggered to this bout. This is
                            only greater than 1 for auto-rewards. 
    '''
    bout_threshold = pgt.get_bout_threshold()

    # For numerical stability, I wipe results and re-annotate 
    if 'bout_number' in session.licks:
        session.licks.drop(columns=['pre_ili','post_ili','rewarded',
            'bout_start','bout_end','bout_number','bout_rewarded',
            'bout_num_rewards','num_rewards'],
            inplace=True,errors='ignore')

    # Remove licks that happen outside of stimulus period
    stim_start=session.stimulus_presentations.query('not omitted')['start_time'].values[0]
    stim_end = session.stimulus_presentations['start_time'].values[-1]+0.75
    session.licks.query('(timestamps > @stim_start) and (timestamps <= @stim_end)',
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
            this_reward_lick_times=np.where(session.licks.timestamps<=row.timestamps)[0]
            if len(this_reward_lick_times) == 0:
                raise Exception('First lick was after first reward')
            else:
                mylick = this_reward_lick_times[-1]
        session.licks.at[mylick,'rewarded'] = True 
        # licks can be double assigned to rewards because of auto-rewards
        session.licks.at[mylick,'num_rewards'] +=1  

    # Annotate bout rewards  
    x = session.licks.groupby('bout_number').any('rewarded').\
        rename(columns={'rewarded':'bout_rewarded'})
    y = session.licks.groupby('bout_number')['num_rewards'].sum().\
        rename(columns={'num_rewards':'bout_num_rewards'})
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
    assert num_rewards == num_lick_rewards+double_rewards, \
        "Lick Annotations don't match number of rewards"

    # Check that all rewards are matched to a bout
    num_rewarded_bouts=np.sum(session.licks['bout_rewarded']&session.licks['bout_start'])
    double_rewarded_bouts = np.sum(session.licks[session.licks['bout_rewarded']&\
        session.licks['bout_start']&\
        (session.licks['bout_num_rewards']>1)]['bout_num_rewards']-1)
    assert num_rewards == num_rewarded_bouts+double_rewarded_bouts, \
        "Bout Annotations don't match number of rewards"
 
    # Check that bouts start and stop
    num_bout_start = session.licks['bout_start'].sum()
    num_bout_end = session.licks['bout_end'].sum()
    num_bouts = session.licks['bout_number'].max()
    assert num_bout_start==num_bout_end, "Bout Starts and Bout Ends don't align"
    assert num_bout_start == num_bouts, "Number of bouts is incorrect"


def annotate_bouts(session):
    '''
        Uses the bout annotations in licks to annotate stimulus_presentations

        Adds to session.stimulus_presentations
            bout_start,     (boolean) Whether a licking bout started during this image
            num_bout_start, (int) The number of licking bouts that started during this
                            image. This can be greater than 1 because the bout duration
                            is less than 750ms. 
            bout_number,    (int) The label of the licking bout that started during this
                            image
            bout_end,       (boolean) Whether a licking bout ended during this image
            num_bout_end,   (int) The number of licking bouts that ended during this
                            image. 

    '''
    # Annotate Bout Starts
    bout_starts = session.licks[session.licks['bout_start']]
    session.stimulus_presentations['bout_start'] = False
    session.stimulus_presentations['num_bout_start'] = 0
    for index,x in bout_starts.iterrows():
        filter_start = session.stimulus_presentations.query('start_time < @x.timestamps')
        if len(filter_start) > 0:
            # Mark the last stimulus that started before the bout started
            start_index = filter_start.index[-1]
            session.stimulus_presentations.at[start_index,'bout_start'] = True
            session.stimulus_presentations.at[start_index,'num_bout_start'] += 1
            session.stimulus_presentations.at[start_index,'bout_number'] = x.bout_number
        elif x.timestamps <= session.stimulus_presentations.iloc[0].start_time:
            # Bout started before stimulus, mark the first stimulus as start
            session.stimulus_presentations.at[0,'bout_start'] = True
            session.stimulus_presentations.at[0,'num_bout_start'] += 1
        else:
            raise Exception('couldnt annotate bout start (bout number: {})'.format(index))

    # Annotate Bout Ends
    bout_ends = session.licks[session.licks['bout_end']]
    session.stimulus_presentations['bout_end'] = False
    session.stimulus_presentations['num_bout_end'] = 0
    for index,x in bout_ends.iterrows():
        filter_end = session.stimulus_presentations.query('start_time < @x.timestamps')
        if len(filter_end) > 0:
            # Mark the last stimulus that started before the bout ended
            end_index = filter_end.index[-1]
            session.stimulus_presentations.at[end_index,'bout_end'] = True
            session.stimulus_presentations.at[end_index,'num_bout_end'] += 1   
        elif x.timestamps <= session.stimulus_presentations.iloc[0].start_time:
            # Bout started before stimulus, mark the first stimulus as start
            session.stimulus_presentations.at[0,'bout_end'] = True
            session.stimulus_presentations.at[0,'num_bout_end'] += 1
        else:
            raise Exception('couldnt annotate bout end (bout number: {})'.format(index))

    # Annotate In-Bout
    session.stimulus_presentations['in_lick_bout'] = \
        session.stimulus_presentations['num_bout_start'].cumsum() > \
        session.stimulus_presentations['num_bout_end'].cumsum()
    session.stimulus_presentations['in_lick_bout'] = \
        session.stimulus_presentations['in_lick_bout'].shift(fill_value=False)
    overlap_index = (session.stimulus_presentations['in_lick_bout']) &\
                    (session.stimulus_presentations['bout_start']) &\
                    (session.stimulus_presentations['num_bout_end'] >=1)
    session.stimulus_presentations.at[overlap_index,'in_lick_bout'] = False

    # QC
    num_bouts_sp_start = session.stimulus_presentations['num_bout_start'].sum()
    num_bouts_sp_end = session.stimulus_presentations['num_bout_end'].sum()
    num_bouts_licks = session.licks.bout_start.sum()
    assert num_bouts_sp_start == num_bouts_licks, \
        "Number of bouts doesnt match between licks table and stimulus table"
    assert num_bouts_sp_start == num_bouts_sp_end, \
        "Mismatch between bout starts and bout ends"
    assert session.stimulus_presentations.query('bout_start')['licked'].all(),\
        "All licking bout start should have licks" 
    assert session.stimulus_presentations.query('bout_end')['licked'].all(),\
        "All licking bout ends should have licks" 
    assert np.all(session.stimulus_presentations['in_lick_bout'] |\
        session.stimulus_presentations['bout_start'] |\
        ~session.stimulus_presentations['licked']), \
        "must either not have licked, or be in lick bout, or bout start"
    assert np.all(~(session.stimulus_presentations['in_lick_bout'] &\
        session.stimulus_presentations['bout_start'])),\
        "Cant be in a bout and a bout_start"


def annotate_image_rolling_metrics(session,win_dur=640, win_type='gaussian',win_std=60):
    '''
        Get rolling image level metrics for lick rate, reward rate, and bout_rate
        Computes over a rolling window of win_dur (s) duration, with a window type 
        given by win_type

        Units are either 1/image or 1/s. We arrive at 1/s by:
        (1/image) * (1 image / 0.75s) = (1/s)

        Adds to session.stimulus_presentations
            num_licks,              (int) number of licks on each image cycle
            lick_rate,              (licks/seconds) 
            rewarded,               (boolean)   Whether this stimulus was rewarded
            reward_rate,            (rewards/seconds)
            bout_rate,              (bouts/seconds)
            hit_bout,               (boolean) Whether this is the start of a rewarded
                                    bout. NaN if not start of lick bout
            lick_hit_fraction,      (%) Percentage of lick bouts that were
                                    rewarded. 
            change_with_lick,       (boolean) Whether this change had a reward
            hit_rate,               (%) Percentage of changes with rewards
            change_without_lick,    (boolean) Whether this change did not have a reward 
            miss_rate,              (%) Percentage of changes without rewards
            non_change_with_lick,   (boolean) Wheter this non-change had a lick
            false_alarm_rate,       (%) Percentage of non-changes with licks
            non_change_without_lick,(boolean) Whether this non-change did not have a lick
            correct_reject_rate,    (%) Percentage of non-changes without licks
            d_prime,                (float)
            criterion,              (float)
            RT,                     (float)
            engaged,                (boolean)
    '''
    # Get Lick Rate / second
    session.stimulus_presentations['num_licks'] = [len(x) for x in \
        session.stimulus_presentations['licks']]
    session.stimulus_presentations['lick_rate'] = \
        session.stimulus_presentations['num_licks'].\
        rolling(win_dur, min_periods=1,win_type=win_type,center=True).\
        mean(std=win_std)/.75

    # Get Reward Rate / second
    session.stimulus_presentations['rewarded'] = [len(this_reward) > 0 \
        for this_reward in session.stimulus_presentations['rewards']]
    session.stimulus_presentations['reward_rate'] = \
        session.stimulus_presentations['rewarded'].\
        rolling(win_dur,min_periods=1,win_type=win_type,center=True).\
        mean(std=win_std)/.75

    # Get Bout Rate / second
    session.stimulus_presentations['bout_rate'] = \
        session.stimulus_presentations['bout_start'].\
        rolling(win_dur,min_periods=1, win_type=win_type,center=True).\
        mean(std=win_std)/.75

    # Get Hit Fraction. % of lick bouts that are rewarded
    session.stimulus_presentations['hit_bout'] = \
        [np.nan if (not x[0]) else 1 if (x[1]==1) else 0 
        for x in zip(session.stimulus_presentations['bout_start'], \
        session.stimulus_presentations['rewarded'])]
    session.stimulus_presentations['lick_hit_fraction'] = \
        session.stimulus_presentations['hit_bout'].\
        rolling(win_dur,min_periods=1,win_type=win_type,center=True).\
        mean(std=win_std).fillna(0)
    
    # Get Hit Rate, % of change images with licks
    session.stimulus_presentations['change_with_lick'] = \
        [np.nan if (not x[0]) else 1 if (x[1]) else 0 
        for x in zip(session.stimulus_presentations['is_change'],\
        session.stimulus_presentations['rewarded'])]
    session.stimulus_presentations['hit_rate'] = \
        session.stimulus_presentations['change_with_lick'].\
        rolling(win_dur, min_periods=1,win_type=win_type,center=True).\
        mean(std=win_std).fillna(0)
  
    # Get Miss Rate, % of change images without licks
    session.stimulus_presentations['change_without_lick'] = \
        [np.nan if (not x[0]) else 0 if (x[1]) else 1 
        for x in zip(session.stimulus_presentations['is_change'],\
        session.stimulus_presentations['rewarded'])]
    session.stimulus_presentations['miss_rate'] = \
        session.stimulus_presentations['change_without_lick'].\
        rolling(win_dur,min_periods=1,win_type=win_type,center=True).\
        mean(std=win_std).fillna(0)

    # Get False Alarm Rate, % of non-change images with licks
    session.stimulus_presentations['non_change_with_lick'] = \
        [np.nan if (x[0]) else 1 if (x[1]) else 0 
        for x in zip(session.stimulus_presentations['is_change'],\
        session.stimulus_presentations['bout_start'])]
    session.stimulus_presentations['false_alarm_rate'] = \
        session.stimulus_presentations['non_change_with_lick'].\
        rolling(win_dur,min_periods=1,win_type=win_type,center=True).\
        mean(std=win_std).fillna(0)

    # Get Correct Reject Rate, % of non-change images without licks
    session.stimulus_presentations['non_change_without_lick'] = \
        [np.nan if (x[0] or (x[2] and not x[1])) else 0 if (x[1]) else 1 
        for x in zip(session.stimulus_presentations['is_change'],\
        session.stimulus_presentations['bout_start'],\
        session.stimulus_presentations['licked'])]
    session.stimulus_presentations['correct_reject_rate'] = \
        session.stimulus_presentations['non_change_without_lick'].\
        rolling(win_dur,min_periods=1,win_type=win_type,center=True).\
        mean(std=win_std).fillna(0)

    # Get dPrime and Criterion metrics on an image level
    # Computing the criterion to be negative
    Z = norm.ppf
    session.stimulus_presentations['d_prime'] = \
        Z(np.clip(session.stimulus_presentations['hit_rate'],0.01,0.99)) - \
        Z(np.clip(session.stimulus_presentations['false_alarm_rate'],0.01,0.99)) 
    session.stimulus_presentations['criterion'] = \
        0.5*(Z(np.clip(session.stimulus_presentations['hit_rate'],0.01,0.99)) + \
        Z(np.clip(session.stimulus_presentations['false_alarm_rate'],0.01,0.99)))
 
    # Add Reaction Time
    session.stimulus_presentations['RT'] = \
        [x[0][0]-x[1] if (len(x[0]) > 0) &x[2] else np.nan \
        for x in zip(session.stimulus_presentations['licks'], \
        session.stimulus_presentations['start_time'], \
        session.stimulus_presentations['bout_start'])]

    # Add engagement classification
    reward_threshold = pgt.get_engagement_threshold()
    session.stimulus_presentations['engaged'] = \
        [x > reward_threshold for x in session.stimulus_presentations['reward_rate']]
    session.stimulus_presentations['engaged_v2'] = \
        [(x[0] > reward_threshold) or (x[1] > .1) for x in \
        zip(session.stimulus_presentations['reward_rate'],\
        session.stimulus_presentations['bout_rate'])]

    # QC
    rewards_sp = session.stimulus_presentations.rewarded.sum()
    licks_sp = session.licks['num_rewards'].sum()
    rewards = len(session.rewards)
    assert rewards_sp == licks_sp, "mismatch between stimulus rewards and lick rewards"
    assert licks_sp == rewards, "mismatch between rewards table and lick rewards"

