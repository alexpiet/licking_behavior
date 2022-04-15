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

def annotate_bout_start_time(session):
    session.stimulus_presentations['bout_start_time'] = np.nan
    session.stimulus_presentations.at[session.stimulus_presentations['bout_start'] == True,'bout_start_time'] = session.stimulus_presentations[session.stimulus_presentations['bout_start']==True].licks.str[0]
    

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
    session.stimulus_presentations['licked'] = [1 if len(this_lick) > 0 else 0 for this_lick in session.stimulus_presentations['licks']]
    session.stimulus_presentations['lick_rate'] = session.stimulus_presentations['licked'].rolling(win_dur, min_periods=1,win_type=win_type).mean()/.75

    # Get Reward Rate / second
    session.stimulus_presentations['rewarded'] = [1 if len(this_reward) > 0 else 0 for this_reward in session.stimulus_presentations['rewards']]
    session.stimulus_presentations['reward_rate'] = session.stimulus_presentations['rewarded'].rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75

    # Get Running / Second
    if add_running:
        session.stimulus_presentations['running_rate'] = session.stimulus_presentations['mean_running_speed'].rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75

    # Get Bout Rate / second
    session.stimulus_presentations['bout_rate'] = session.stimulus_presentations['bout_start'].rolling(win_dur,min_periods=1, win_type=win_type).mean()/.75

    # Get Hit Fraction. % of licks that are rewarded
    session.stimulus_presentations['hit_bout'] = [np.nan if (not x[0]) else 1 if (x[1]==1) else 0 for x in zip(session.stimulus_presentations['bout_start'], session.stimulus_presentations['rewarded'])]
    session.stimulus_presentations['lick_hit_fraction'] = session.stimulus_presentations['hit_bout'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)
    
    # Get Hit Rate, % of change flashes with licks
    session.stimulus_presentations['change_with_lick'] = [np.nan if (not x[0]) else 1 if (x[1]) else 0 for x in zip(session.stimulus_presentations['change'],session.stimulus_presentations['bout_start'])]
    session.stimulus_presentations['hit_rate'] = session.stimulus_presentations['change_with_lick'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)
  
    # Get Miss Rate, % of change flashes without licks
    session.stimulus_presentations['change_without_lick'] = [np.nan if (not x[0]) else 0 if (x[1]) else 1 for x in zip(session.stimulus_presentations['change'],session.stimulus_presentations['bout_start'])]
    session.stimulus_presentations['miss_rate'] = session.stimulus_presentations['change_without_lick'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)

    # Get False Alarm Rate, % of non-change flashes with licks
    session.stimulus_presentations['non_change_with_lick'] = [np.nan if (x[0]) else 1 if (x[1]) else 0 for x in zip(session.stimulus_presentations['change'],session.stimulus_presentations['bout_start'])]
    session.stimulus_presentations['false_alarm_rate'] = session.stimulus_presentations['non_change_with_lick'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)

    # Get Correct Reject Rate, % of non-change flashes without licks
    session.stimulus_presentations['non_change_without_lick'] = [np.nan if (x[0]) else 0 if (x[1]) else 1 for x in zip(session.stimulus_presentations['change'],session.stimulus_presentations['bout_start'])]
    session.stimulus_presentations['correct_reject_rate'] = session.stimulus_presentations['non_change_without_lick'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)

    # Get dPrime and Criterion metrics on a flash level
    Z = norm.ppf
    session.stimulus_presentations['d_prime']   = Z(np.clip(session.stimulus_presentations['hit_rate'],0.01,0.99)) - Z(np.clip(session.stimulus_presentations['false_alarm_rate'],0.01,0.99)) 
    session.stimulus_presentations['criterion'] = 0.5*(Z(np.clip(session.stimulus_presentations['hit_rate'],0.01,0.99)) + Z(np.clip(session.stimulus_presentations['false_alarm_rate'],0.01,0.99)))
        # Computing the criterion to be negative
    
    # Add Reaction Time
    session.stimulus_presentations['RT'] = [x[0][0]-x[1] if (len(x[0]) > 0) &x[2] else np.nan for x in zip(session.stimulus_presentations['licks'], session.stimulus_presentations['start_time'], session.stimulus_presentations['bout_start'])]
 
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


'''
Functions below here are for plotting and analysis, not computation

The first set of functions is for single session analysis

'''
def plot_all_metrics(manifest,verbose=False):
    # make session plots for all sessions
    ids = manifest['behavior_session_id'].values
    num_crashed =0
    for id in tqdm(ids):
        try:
            filename = MODEL_FREE_DIR+'session_figures/'+str(id)
            session = pgt.get_data(id)
            get_metrics(session)
            plot_metrics(session,filename=filename+'_metrics')
            plt.close('all')
        except:
            num_crashed += 1
            if verbose:
                print(f"{id} crash")
    print(str(num_crashed) +' sessions crashed')
    print(str(len(ids) - num_crashed) + ' sessions saved')


def plot_metrics_from_table(df, iloc):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(11.5,5))

    cluster_labels = df.iloc[iloc].engaged
    cluster_labels=[0 if x else 1 for x in cluster_labels]
    colors = pstyle.get_project_colors()
    style = pstyle.get_style()
    cp = np.where(~(np.diff(cluster_labels) == 0))[0]
    cp = np.concatenate([[0], cp, [len(cluster_labels)]])
    plotted = np.zeros(2,)
    labels = ['engaged','disengaged']
    for i in range(0, len(cp)-1):
        if plotted[cluster_labels[cp[i]+1]]:
            ax.axvspan(cp[i],cp[i+1],edgecolor=None,facecolor=colors[labels[cluster_labels[cp[i]+1]]], alpha=0.2)
        else:
            plotted[cluster_labels[cp[i]+1]] = True
            ax.axvspan(cp[i],cp[i+1],edgecolor=None,facecolor=colors[labels[cluster_labels[cp[i]+1]]], alpha=0.2,label=labels[cluster_labels[cp[i]+1]])

    ax.plot(df.iloc[iloc].reward_rate,'m',label='Reward Rate')
    ax.axhline(1/90,linestyle='--',alpha=0.5,color='m',label='Engagement Threshold')

    ax.plot(df.iloc[iloc].lick_bout_rate,'g',label='Lick Rate')
    ax.set_xlabel('Image #',fontsize=style['label_fontsize'])
    ax.set_ylabel('rate/sec',fontsize=style['label_fontsize'])
    ax.tick_params(axis='both',labelsize=style['axis_ticks_fontsize'])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim([0,len(df.iloc[iloc].engaged)])
    ax.set_ylim([0,.5])
    plt.tight_layout()



def plot_metrics(session):
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(11.5,5))
    if 'bout_rate' not in session.stimulus_presentations:
        annotate_flash_rolling_metrics(session)
        classify_by_flash_metrics(session)

    cluster_labels = session.stimulus_presentations['engaged'].values
    cluster_labels=[0 if x else 1 for x in cluster_labels]
    colors = pstyle.get_project_colors()
    style = pstyle.get_style()
    cp = np.where(~(np.diff(cluster_labels) == 0))[0]
    cp = np.concatenate([[0], cp, [len(cluster_labels)]])
    plotted = np.zeros(2,)
    labels = ['engaged','disengaged']
    for i in range(0, len(cp)-1):
        if plotted[cluster_labels[cp[i]+1]]:
            ax.axvspan(cp[i],cp[i+1],edgecolor=None,facecolor=colors[labels[cluster_labels[cp[i]+1]]], alpha=0.2)
        else:
            plotted[cluster_labels[cp[i]+1]] = True
            ax.axvspan(cp[i],cp[i+1],edgecolor=None,facecolor=colors[labels[cluster_labels[cp[i]+1]]], alpha=0.2,label=labels[cluster_labels[cp[i]+1]])

    ax.plot(session.stimulus_presentations.reward_rate,'m',label='Reward Rate')
    ax.axhline(1/90,linestyle='--',alpha=0.5,color='m',label='Engagement Threshold')

    ax.plot(session.stimulus_presentations.bout_rate,'g',label='Lick Rate')
    ax.set_xlabel('Image #',fontsize=style['label_fontsize'])
    ax.set_ylabel('rate/sec',fontsize=style['label_fontsize'])
    ax.tick_params(axis='both',labelsize=style['axis_ticks_fontsize'])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlim([0,len(session.stimulus_presentations)])
    ax.set_ylim([0,.5])
    plt.tight_layout()


 
def plot_metrics_old(session,use_bouts=True,filename=None):
    '''
        plot the lick and reward rates for this session with the classified epochs
        over the course of the session
    '''
    fig,ax = plt.subplots(nrows=3,ncols=1,figsize=(10,8))
    if 'bout_rate' not in session.stimulus_presentations:
        annotate_flash_rolling_metrics(session)
        classify_by_flash_metrics(session)
    
    cluster_labels = session.stimulus_presentations['flash_metrics_epochs'].values
    cluster_colors = sns.color_palette(n_colors=3)
    cluster_colors = np.vstack([cluster_colors[1], cluster_colors[0],cluster_colors[2]])
    cp = np.where(~(np.diff(cluster_labels) == 0))[0]
    cp = np.concatenate([[0], cp, [len(cluster_labels)]])
    plotted = np.zeros(3,)
    labels = ['low-lick, low-reward','high-lick, high-reward','high-lick, low-reward']
    for i in range(0, len(cp)-1):
        if plotted[cluster_labels[cp[i]+1]]:
            ax[0].axvspan(cp[i],cp[i+1],color=cluster_colors[cluster_labels[cp[i]+1]], alpha=0.2)
        else:
            plotted[cluster_labels[cp[i]+1]] = True
            ax[0].axvspan(cp[i],cp[i+1],color=cluster_colors[cluster_labels[cp[i]+1]], alpha=0.2,label=labels[cluster_labels[cp[i]+1]])

    ax[0].plot(session.stimulus_presentations.reward_rate,'m',label='Reward Rate')
    ax[0].axhline(2/80,linestyle='--',alpha=0.5,color='m',label='Reward Threshold')
    if use_bouts:
        ax[0].plot(session.stimulus_presentations.bout_rate,'g',label='Lick Rate')
    else:
        ax[0].plot(session.stimulus_presentations.lick_rate,'g',label='Flash Lick')
    ax[0].axhline(.1,linestyle='--',alpha=0.5,color='g',label='Lick Threshold')
    ax[0].set_xlabel('Flash #',fontsize=16)
    ax[0].set_ylabel('Rate/Flash',fontsize=16)
    ax[0].tick_params(axis='both',labelsize=12)
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[0].set_xlim([0,len(session.stimulus_presentations)])
    ax[0].set_ylim([0,1])


    ax[1].plot(session.stimulus_presentations.bout_rate,'g',label='Lick Rate')
    ax[1].plot(session.stimulus_presentations.lick_hit_fraction,'b',label='Lick Hit Fraction')
    ax[1].plot(session.stimulus_presentations.hit_rate,'r',label='Hit Rate')
    ax[1].plot(session.stimulus_presentations.false_alarm_rate,'k',label='False Alarm')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].set_xlim([0,len(session.stimulus_presentations)])
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel('Flash #',fontsize=16)
    ax[1].set_ylabel('Rate',fontsize=16)
    ax[1].tick_params(axis='both',labelsize=12)

    ax[2].plot(session.stimulus_presentations.d_prime,'k',label='d prime')
    ax[2].plot(session.stimulus_presentations.criterion,'r',label='criterion')
    ax[2].axhline(0,linestyle='--',alpha=0.5,color='k')
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[2].set_xlim([0,len(session.stimulus_presentations)])
    ax[2].set_ylim(bottom=-1)
    ax[2].set_xlabel('Flash #',fontsize=16)
    ax[2].set_ylabel('d prime',fontsize=16)
    ax[2].tick_params(axis='both',labelsize=12)

    plt.tight_layout()   
    if type(filename) is not None:
        plt.savefig(filename+".png")
 
# UPDATE REQUIRED
def plot_2D(session,lick_threshold = 0.1, reward_threshold = 2/80,filename=None):
    '''
        plot the lick and reward rates for this session with the classified epochs
        in 2D space
    '''
    plt.figure()
    if 'bout_rate' not in session.stimulus_presentations:
        annotate_flash_rolling_metrics(session)  
    cluster_colors = sns.color_palette(n_colors=3)  
    cluster_colors = np.vstack([cluster_colors[1], cluster_colors[0],cluster_colors[2]])
    patch1 = patches.Rectangle((0,0),reward_threshold,lick_threshold,edgecolor=cluster_colors[0],facecolor=cluster_colors[0],alpha=0.2)
    plt.gca().add_patch(patch1)
    patch2 = patches.Rectangle((reward_threshold,0),0.05,1,edgecolor=cluster_colors[1],facecolor=cluster_colors[1],alpha=0.2)
    plt.gca().add_patch(patch2)
    patch3 = patches.Rectangle((0,lick_threshold),reward_threshold,1,edgecolor=cluster_colors[2],facecolor=cluster_colors[2],alpha=0.2)
    plt.gca().add_patch(patch3)
    plt.plot(session.stimulus_presentations.reward_rate, session.stimulus_presentations.bout_rate,'ko',alpha=.1)
    plt.ylim([0, 0.4])
    plt.plot([0,reward_threshold],[lick_threshold, lick_threshold],linestyle='--',color='r',alpha=0.5)
    plt.axvline(reward_threshold,linestyle='--',color='r',alpha=0.5)
    plt.xlim(xmin=0)
    plt.ylabel('lick rate/flash')
    plt.xlabel('reward rate/flash')
    if type(filename) is not type(None):
        plt.savefig(filename+".png")

'''
    Functions below here are for population analysis
'''
def plot_rates_summary(df,group=None):
    plot_rates(df, ['d_prime'],group=group)
    plot_rates(df, ['hit_rate'],group=group)
    plot_rates(df, ['fa_rate'],group=group)
    plot_rates(df, ['lick_bout_rate'],group=group)
    plot_rates(df, ['criterion'],group=group)
    plot_rates(df, ['reward_rate'],group=group)
    plot_rates(df, ['engaged'],group=group)
    plot_rates(df, ['lick_hit_fraction'],group=group)
    plot_rates(df, ['d_prime','criterion'], group=group,label='d_prime_and_criterion')
    plot_rates(df, ['lick_bout_rate','reward_rate'], group=group,label='lick_rate')
    plot_rates(df, ['hit_rate','fa_rate'], group=group,label='hit_rates')
    #plot_rates(df, ['low_lick_low_reward','high_lick_high_reward','high_lick_low_reward'],group=group, label='state')

def plot_counts_summary(df,group=None):
    plot_counts(df, ['num_hits'],group=group,ylim=(0,None))
    plot_counts(df, ['num_trials'],group=group,ylim=(0,None))
    plot_counts(df, ['d_prime_avg'],group=group,ylim=(0,None))
    plot_counts(df, ['hit_rate_avg'],group=group,ylim=(0,None))
    plot_counts(df, ['fa_rate_avg'],group=group,ylim=(0,None))
    plot_counts(df, ['lick_bout_rate_avg'],group=group,ylim=(0,None))
    plot_counts(df, ['criterion_avg'],group=group)
    plot_counts(df, ['reward_rate_avg'],group=group,ylim=(0,None))
    plot_counts(df, ['lick_hit_fraction_avg'],group=group,ylim=(0,None))
    plot_counts(df, ['fraction_engaged'],group=group,ylim=(0,1))
    #plot_counts(df, ['fraction_low_lick_low_reward'], group=group,ylim=(0,1))
    #plot_counts(df, ['fraction_high_lick_low_reward'], group=group,ylim=(0,1))
    #plot_counts(df, ['fraction_high_lick_high_reward'], group=group,ylim=(0,1))
    #plot_counts(df, ['fraction_low_lick_low_reward','fraction_high_lick_high_reward','fraction_high_lick_low_reward'], group=group,ylim=(0,1),label='epoch')
    plot_counts(df, ['fraction_engaged_1st','fraction_engaged_2nd'], label='engaged_by_half',group=group, ylim=(0,1))
    #plot_counts(df, ['fraction_low_lick_low_reward_1st','fraction_low_lick_low_reward_2nd'], label='low_lick_low_reward_by_half',group=group, ylim=(0,1))
    #plot_counts(df, ['fraction_high_lick_high_reward_1st','fraction_high_lick_high_reward_2nd'], label='high_lick_high_reward_by_half',group=group, ylim=(0,1))
    #plot_counts(df, ['fraction_high_lick_low_reward_1st','fraction_high_lick_low_reward_2nd'], label='high_lick_low_reward_by_half',group=group, ylim=(0,1))
    plot_counts(df, ['d_prime_1st','d_prime_2nd'],group=group,label='dprime_by_half')
    plot_counts(df, ['hit_rate_1st','hit_rate_2nd'],group=group,label='hit_rate_by_half')
    plot_counts(df, ['fa_rate_1st','fa_rate_2nd'],group=group,label='fa_rate_by_half')
    plot_counts(df, ['lick_bout_rate_1st','lick_bout_rate_2nd'],group=group,label='lick_bout_rate_by_half')
    plot_counts(df, ['criterion_1st','criterion_2nd'],group=group,label='criterion_by_half')
    plot_counts(df, ['reward_rate_1st','reward_rate_2nd'],group=group,label='reward_rate_by_half')
    plot_counts(df, ['lick_hit_fraction_1st','lick_hit_fraction_2nd'],group=group,label='lick_hit_fraction_by_half')

def get_colors():
    tab10= plt.get_cmap("tab10")
    colors = {
        'd_prime':'r',
        'hit_rate':'g',
        'fa_rate':'k',
        'lick_bout_rate':'r',
        'criterion':'b',
        'reward_rate':'r',
        'engaged':'r',
        'lick_hit_fraction':'r',
        'low_lick_low_reward':tab10(0),
        'high_lick_low_reward':tab10(1),
        'high_lick_high_reward':tab10(2),
        'fraction_low_lick_low_reward':tab10(0),
        'fraction_high_lick_low_reward':tab10(1),
        'fraction_high_lick_high_reward':tab10(2),
        'low_lick_low_reward_1st':tab10(0),
        'high_lick_low_reward_1st':tab10(1),
        'high_lick_high_reward_1st':tab10(2),
        'fraction_low_lick_low_reward_1st':tab10(0),
        'fraction_high_lick_low_reward_1st':tab10(1),
        'fraction_high_lick_high_reward_1st':tab10(2),
        'low_lick_low_reward_2nd':tab10(0),
        'high_lick_low_reward_2nd':tab10(1),
        'high_lick_high_reward_2nd':tab10(2),
        'fraction_low_lick_low_reward_2nd':tab10(0),
        'fraction_high_lick_low_reward_2nd':tab10(1),
        'fraction_high_lick_high_reward_2nd':tab10(2),
        'Sst-IRES-Cre' : (158/255,218/255,229/255),
        'Vip-IRES-Cre' : (197/255,176/255,213/255),
        'Slc17a7-IRES2-Cre' : (255/255,152/255,150/255),
         'OPHYS_1_images_A':(148/255,29/255,39/255),
         'OPHYS_2_images_A':(222/255,73/255,70/255),
         'OPHYS_3_images_A':(239/255,169/255,150/255),
         'OPHYS_4_images_A':(43/255,80/255,144/255),
         'OPHYS_5_images_A':(100/255,152/255,193/255),
         'OPHYS_6_images_A':(195/255,216/255,232/255),
         'OPHYS_1_images_B':(148/255,29/255,39/255),
         'OPHYS_2_images_B':(222/255,73/255,70/255),
         'OPHYS_3_images_B':(239/255,169/255,150/255),
         'OPHYS_4_images_B':(43/255,80/255,144/255),
         'OPHYS_5_images_B':(100/255,152/255,193/255),
         'OPHYS_6_images_B':(195/255,216/255,232/255),
         'F1':(148/255,29/255,39/255),
         'F2':(222/255,73/255,70/255),
         'F3':(239/255,169/255,150/255),
         'N1':(43/255,80/255,144/255),
         'N2':(100/255,152/255,193/255),
         'N3':(195/255,216/255,232/255)

    }
    return colors

def get_clean_label():
    type_dict = {
        'OPHYS_1_images_A':'F1',
        'OPHYS_2_images_A':'F2',
        'OPHYS_3_images_A':'F3',
        'OPHYS_4_images_A':'N1',
        'OPHYS_5_images_A':'N2',
        'OPHYS_6_images_A':'N3',
        'OPHYS_1_images_B':'F1',
        'OPHYS_2_images_B':'F2',
        'OPHYS_3_images_B':'F3',
        'OPHYS_4_images_B':'N1',
        'OPHYS_5_images_B':'N2',
        'OPHYS_6_images_B':'N3',
        'Sst-IRES-Cre':'Sst',
        'Vip-IRES-Cre':'Vip',
        'Slc17a7-IRES2-Cre':'Slc',
        'fraction_low_lick_low_reward':'low lick,\n low reward', 
        'fraction_high_lick_high_reward':'high lick,\n high reward', 
        'fraction_high_lick_low_reward':'high lick,\n low reward',
        'fraction_low_lick_low_reward_1st':'low lick, low reward\n 1st', 
        'fraction_high_lick_high_reward_1st':'high lick, high reward\n 1st', 
        'fraction_high_lick_low_reward_1st':'high lick, low reward\n 1st',
        'fraction_low_lick_low_reward_2nd':'low lick, low reward\n 2nd', 
        'fraction_high_lick_high_reward_2nd':'high lick, high reward\n 2nd', 
        'fraction_high_lick_low_reward_2nd':'high lick, low reward\n 2nd'
    }
    return type_dict

def get_styles():
    styles = {
        'Sst-IRES-Cre':'--',
        'Vip-IRES-Cre':'-',
        'Slc17a7-IRES2-Cre':'-.',
        'OPHYS_1_images_A':'--',
        'OPHYS_3_images_A':'--',
        'OPHYS_4_images_A':'--',
        'OPHYS_6_images_A':'--',
        'OPHYS_1_images_B':'--',
        'OPHYS_3_images_B':'--',
        'OPHYS_4_images_B':'--',
        'OPHYS_6_images_B':'--'
    }
    return styles

def plot_counts(df, counts, group=None, label=None,ylim=None,fs1=16,fs2=14,xlabel=None,label_dict=None):
    if (len(counts) > 1) & (group is not None):
        plt.figure(figsize=(10,5))
    else:
        plt.figure(figsize=(5,5))
    colors = get_colors()
    styles = get_styles()
    clean_labels = get_clean_label() 
    if group is not None:
        groups = df[group].unique()
    labels = []
    xloc = 0
    for cdex, count in enumerate(counts):
        if group is not None:
            for index, g in enumerate(groups):
                g_df = df[df[group] == g].copy()
                val = np.nanmean(g_df[count])
                sem = np.std(g_df[count])/np.sqrt(len(g_df))
                plt.plot([xloc-.45,xloc+.45],[val,val],'-',linewidth=4,color=colors.get(g,'k'))
                plt.plot([xloc,xloc], [val-sem, val+sem],'k-',alpha=.5)
                if len(counts) >1:
                    labels.append(clean_labels.get(count,count)+' '+clean_labels.get(g,g))               
                else:
                    labels.append(clean_labels.get(g,g))
                xloc+=1
        else:
            val = np.nanmean(df[count])
            sem = np.std(df[count])/np.sqrt(len(df))
            plt.plot([xloc-.45,xloc+.45],[val,val],'-',linewidth=4,color=colors.get(count,'k'))
            plt.plot([xloc,xloc], [val-sem, val+sem],'k-',alpha=.5)
            labels.append(clean_labels.get(count,count))
            xloc+=1

    if len(counts) == 1:
        if label is None:
            plt.ylabel(counts[0],fontsize=fs1)
        else:
            plt.ylabel(label,fontsize=fs1)
        if xlabel is None:
            plt.xlabel(group,fontsize=fs1)
        else:
            plt.xlabel(xlabel,fontsize=fs1)
        if label_dict is not None:
            labels = [label_dict[x] for x in labels]
        plt.xticks(np.arange(0,xloc), labels,fontsize=fs2)
    else:
        plt.ylabel(label,fontsize=fs1)
        if label_dict is not None:
            labels = [label_dict[x] for x in labels]
        plt.xticks(np.arange(0,xloc), labels,fontsize=fs2,rotation=90)   
    plt.yticks(fontsize=fs1)
    if ylim is not None:
        plt.ylim(ylim)
    plt.tight_layout()
    if group is not None:
        group = '_by_'+str(group)
    else:
        group =''
    if len(counts) == 1:
        label = counts[0]
    plt.savefig(MODEL_FREE_DIR+'summary_figures/avg_'+label+group+'.png')
    plt.savefig(MODEL_FREE_DIR+'summary_figures/avg_'+label+group+'.svg')
 
def plot_rates(df, rates, group=None,label=None,fs1=16,fs2=14,legends=None):
    plt.figure(figsize=(10,5))
    colors = get_colors()
    styles = get_styles()
    if group is not None:
        groups = df[group].unique()
    for rate in rates:
        if group is not None:
            for g in groups:
                print(rate + '-'+g)
                g_df = df[df[group] == g].copy()
                if legends is None:
                    plt.plot(np.nanmean(np.vstack(g_df[rate]),axis=0),color=colors.get(g,'k'),label=g+' '+rate,linestyle=styles.get(g,'-'))
                else:
                    plt.plot(np.nanmean(np.vstack(g_df[rate])*100,axis=0),color=colors.get(g,'k'),label=legends[g],linestyle=styles.get(g,'-'),linewidth=4)
        else:
            plt.plot(np.nanmean(np.vstack(df[rate]),axis=0),color=colors.get(rate,'k'),label=rate)
    if (label is None) &(len(rates) ==1):
        label = rates[0]
    plt.ylabel(label,fontsize=fs1)
    plt.xlabel('Image #',fontsize=fs1)
    plt.axhline(0, color='k',linestyle='--',alpha=.3)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.xlim(0,4800)
    plt.ylim(bottom=0)
    plt.legend(fontsize=fs2)
    plt.tight_layout()
    if group is not None:
        group = '_by_'+str(group)
    else:
        group =''
    plt.savefig(MODEL_FREE_DIR+'summary_figures/avg_'+label+group+'.png')
    plt.savefig(MODEL_FREE_DIR+'summary_figures/avg_'+label+group+'.svg')

def build_metrics_df(TRAIN=False):
    if TRAIN:
        manifest = pgt.get_training_manifest()
    else:
        manifest = pgt.get_ophys_manifest()

    # Add columns
    crashed = 0
    manifest['metrics_available'] = manifest['active'] # copying size
    columns = {'lick_bout_rate','reward_rate','hit_rate','lick_hit_fraction','fa_rate','d_prime','criterion','engaged',} 
    for column in columns:
        manifest[column] = [[]]*len(manifest)
    for index, row in tqdm(manifest.iterrows(), total = manifest.shape[0]):
        try:
            session = pgt.get_data(row.behavior_session_id)
            get_metrics(session)
        except:
            if verbose:
                print(str(row.behavior_session_id) + ' crashed')
            manifest.at[index,'metrics_availabile'] = False
            crashed +=1
        else:
            manifest.at[index,'lick_bout_rate'] = pgt.get_clean_rate(session.stimulus_presentations['bout_rate'].values)
            manifest.at[index,'reward_rate']    = pgt.get_clean_rate(session.stimulus_presentations['reward_rate'].values)
            manifest.at[index,'hit_rate']       = pgt.get_clean_rate(session.stimulus_presentations['hit_rate'].values)
            manifest.at[index,'lick_hit_fraction']   = pgt.get_clean_rate(session.stimulus_presentations['lick_hit_fraction'].values) 
            manifest.at[index,'fa_rate']        = pgt.get_clean_rate(session.stimulus_presentations['false_alarm_rate'].values)
            manifest.at[index,'d_prime']        = pgt.get_clean_rate(session.stimulus_presentations['d_prime'].values)
            manifest.at[index,'criterion']      = pgt.get_clean_rate(session.stimulus_presentations['criterion'].values)
            #manifest.at[index,'flash_metrics_epochs'] = pgt.get_clean_rate(session.stimulus_presentations['flash_metrics_epochs'].values)
            manifest.at[index,'engaged']        = [(x==1) or (x==2) for x in manifest.at[index,'flash_metrics_epochs']]
            manifest.at[index,'num_hits']       = np.sum(session.trials.hit)
            manifest.at[index,'num_trials']     = len(session.trials)
            #manifest.at[index,'fraction_low_lick_low_reward']   = np.sum(manifest.at[index,'flash_metrics_epochs'] == 0)/4800
            #manifest.at[index,'fraction_high_lick_high_reward'] = np.sum(manifest.at[index,'flash_metrics_epochs'] == 1)/4800    
            #manifest.at[index,'fraction_high_lick_low_reward']  = np.sum(manifest.at[index,'flash_metrics_epochs'] == 2)/4800   
            manifest.at[index,'fraction_engaged']               = np.sum(manifest.at[index,'engaged'])/4800

    if TRAIN:
        manifest.to_pickle(MODEL_FREE_DIR+'psy_metrics_manifest_march_2021_release_training.pkl')
    else:
        manifest.to_pickle(MODEL_FREE_DIR+'psy_metrics_manifest_march_2021_release.pkl')   
    return manifest
    
def get_metrics_df(TRAIN=False,split=2400):
    if TRAIN:
        manifest = pd.read_pickle(MODEL_FREE_DIR+'psy_metrics_manifest_march_2021_release_training.pkl')
    else:
        manifest = pd.read_pickle(MODEL_FREE_DIR+'psy_metrics_manifest_march_2021_release.pkl')      
    manifest.rename({'hit_fraction':'lick_hit_fraction'},axis=1, inplace=True,errors='ignore') 
    #manifest['low_lick_low_reward']   = [x ==0 for x in manifest['flash_metrics_epochs']]
    #manifest['high_lick_high_reward'] = [x ==1 for x in manifest['flash_metrics_epochs']]
    #manifest['high_lick_low_reward']  = [x ==2 for x in manifest['flash_metrics_epochs']]
    type_dict = get_clean_label()
    manifest['session_type'] = [type_dict[x] for x in manifest['session_type']]
    #manifest['fraction_low_lick_low_reward_1st'] = [np.nanmean(x[0:split]) for x in manifest['low_lick_low_reward']]
    #manifest['fraction_low_lick_low_reward_2nd'] = [np.nanmean(x[split:]) for x in manifest['low_lick_low_reward']]
    #manifest['fraction_high_lick_low_reward_1st'] = [np.nanmean(x[0:split]) for x in manifest['high_lick_low_reward']]
    #manifest['fraction_high_lick_low_reward_2nd'] = [np.nanmean(x[split:]) for x in manifest['high_lick_low_reward']]
    #manifest['fraction_high_lick_high_reward_1st'] = [np.nanmean(x[0:split]) for x in manifest['high_lick_high_reward']]
    #manifest['fraction_high_lick_high_reward_2nd'] = [np.nanmean(x[split:]) for x in manifest['high_lick_high_reward']]
    manifest['fraction_engaged_avg'] = [np.nanmean(x) for x in manifest['engaged']]
    manifest['d_prime_avg'] = [np.nanmean(x) for x in manifest['d_prime']]
    manifest['fa_rate_avg'] = [np.nanmean(x) for x in manifest['fa_rate']]
    manifest['lick_bout_rate_avg'] = [np.nanmean(x) for x in manifest['lick_bout_rate']]
    manifest['criterion_avg'] = [np.nanmean(x) for x in manifest['criterion']]
    manifest['reward_rate_avg'] = [np.nanmean(x) for x in manifest['reward_rate']]
    manifest['lick_hit_fraction_avg'] = [np.nanmean(x) for x in manifest['lick_hit_fraction']]
    manifest['hit_rate_avg'] = [np.nanmean(x) for x in manifest['hit_rate']]
    manifest['fraction_engaged_1st'] = [np.nanmean(x[0:split]) for x in manifest['engaged']]
    manifest['fraction_engaged_2nd'] = [np.nanmean(x[split:]) for x in manifest['engaged']]
    manifest['d_prime_1st'] = [np.nanmean(x[0:split]) for x in manifest['d_prime']]
    manifest['d_prime_2nd'] = [np.nanmean(x[split:]) for x in manifest['d_prime']]
    manifest['fa_rate_1st'] = [np.nanmean(x[0:split]) for x in manifest['fa_rate']]
    manifest['fa_rate_2nd'] = [np.nanmean(x[split:]) for x in manifest['fa_rate']]
    manifest['lick_bout_rate_1st'] = [np.nanmean(x[0:split]) for x in manifest['lick_bout_rate']]
    manifest['lick_bout_rate_2nd'] = [np.nanmean(x[split:]) for x in manifest['lick_bout_rate']]
    manifest['criterion_1st'] = [np.nanmean(x[0:split]) for x in manifest['criterion']]
    manifest['criterion_2nd'] = [np.nanmean(x[split:]) for x in manifest['criterion']]
    manifest['reward_rate_1st'] = [np.nanmean(x[0:split]) for x in manifest['reward_rate']]
    manifest['reward_rate_2nd'] = [np.nanmean(x[split:]) for x in manifest['reward_rate']]
    manifest['lick_hit_fraction_1st'] = [np.nanmean(x[0:split]) for x in manifest['lick_hit_fraction']]
    manifest['lick_hit_fraction_2nd'] = [np.nanmean(x[split:]) for x in manifest['lick_hit_fraction']]
    manifest['hit_rate_1st'] = [np.nanmean(x[0:split]) for x in manifest['hit_rate']]
    manifest['hit_rate_2nd'] = [np.nanmean(x[split:]) for x in manifest['hit_rate']]   
    return manifest

def plot_engagement_landscape(df,plot_threshold=True):
    style = pstyle.get_style()
    lick_bout_rate = np.concatenate(df['lick_bout_rate'].values)
    reward_rate = np.concatenate(df['reward_rate'].values)
    lick_bout_rate = lick_bout_rate[~np.isnan(lick_bout_rate)]
    reward_rate = reward_rate[~np.isnan(reward_rate)]

    plt.figure(figsize=(5,5))
    h= plt.hist2d(lick_bout_rate, reward_rate, bins=100,cmax=5000,cmap='magma')
    #plt.gcf().colorbar(h[3],ax=plt.gca())
    plt.xlabel('Lick Bout Rate (bouts/sec)',fontsize=style['label_fontsize'])
    plt.ylabel('Reward Rate (rewards/sec)',fontsize=style['label_fontsize'])
    plt.ylim(top=.10)
    plt.xlim(right=.5)
    plt.gca().tick_params(axis='both',labelsize=style['axis_ticks_fontsize'])
    plt.tight_layout()
    if plot_threshold:
        #plt.arrow(-.1,2/80,.1,0,color='red',zorder=100000)
        plt.gca().annotate('',xy=(0,1/90),xycoords='data',xytext=(-.05,1/90),arrowprops=dict(arrowstyle='->',color='r',lw=1.5))
        plt.gca().annotate('',xy=(.5,1/90),xycoords='data',xytext=(.55,1/90),arrowprops=dict(arrowstyle='->',color='r',lw=1.5))
        #plt.plot([0,.1],[2/80,2/80], 'g')
        #plt.plot([.1,.1],[0,2/80], 'g')
        #rect = patches.Rectangle((0,0),.1,2/80,color='g', alpha=.5)
        #plt.gca().add_patch(rect)
        plt.savefig(MODEL_FREE_DIR+'summary_figures/engagement_landscape_threshold.png')
        plt.savefig(MODEL_FREE_DIR+'summary_figures/engagement_landscape_threshold.svg')
    else:
        plt.savefig(MODEL_FREE_DIR+'summary_figures/engagement_landscape.png')
        plt.savefig(MODEL_FREE_DIR+'summary_figures/engagement_landscape.svg')



