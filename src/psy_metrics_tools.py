import numpy as np
import matplotlib.pyplot as plt
import psy_general_tools as pgt
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
import scipy.stats as ss
from scipy.stats import norm
from scipy import stats

'''
This is a set of functions for calculating and analyzing model free behavioral metrics on a flash by flash basis
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
    annotate_flash_rolling_metrics(session)
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
    

def annotate_flash_rolling_metrics(session,win_dur=320, win_type='triang'):
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
    session.stimulus_presentations['running_rate'] = session.stimulus_presentations['mean_running_speed'].rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75

    # Get Bout Rate / second
    session.stimulus_presentations['bout_rate'] = session.stimulus_presentations['bout_start'].rolling(win_dur,min_periods=1, win_type=win_type).mean()/.75

    # Get Hit Fraction. % of licks that are rewarded
    session.stimulus_presentations['hit_bout'] = [np.nan if (not x[0]) else 1 if (x[1]==1) else 0 for x in zip(session.stimulus_presentations['bout_start'], session.stimulus_presentations['rewarded'])]
    session.stimulus_presentations['hit_fraction'] = session.stimulus_presentations['hit_bout'].rolling(win_dur,min_periods=1,win_type=win_type).mean().fillna(0)
    
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

    Z = norm.ppf
    session.stimulus_presentations['d_prime'] = Z(np.clip(session.stimulus_presentations['hit_rate'],0.01,0.99)) - Z(np.clip(session.stimulus_presentations['false_alarm_rate'],0.01,0.99)) 

 
def classify_by_flash_metrics(session, lick_threshold = 0.1, reward_threshold=2/80,use_bouts=True):
    '''
        Use the flash level rolling metrics to classify into three states based on the thresholds
        lick_threshold is the licking rate / flash that divides high and low licking states
        reward_threshold is the rewards/flash that divides high and low reward states (2/80 is equivalent to 2 rewards/minute). 
    '''
    if use_bouts:
        session.stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in session.stimulus_presentations['bout_rate']] 
    else:
        session.stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in session.stimulus_presentations['lick_rate']] 
    session.stimulus_presentations['high_reward'] = [True if x > reward_threshold else False for x in session.stimulus_presentations['reward_rate']] 
    session.stimulus_presentations['flash_metrics_epochs'] = [0 if (not x[0]) & (not x[1]) else 1 if x[1] else 2 for x in zip(session.stimulus_presentations['high_lick'], session.stimulus_presentations['high_reward'])]
    session.stimulus_presentations['flash_metrics_labels'] = ['low-lick,low-reward' if x==0  else 'high-lick,high-reward' if x==1 else 'high-lick,low-reward' for x in session.stimulus_presentations['flash_metrics_epochs']]


'''
Functions below here are for plotting and analysis, not computation

The first set of functions is for single session analysis

'''
def plot_metrics(session,use_bouts=True,filename=None):
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
    ax[1].plot(session.stimulus_presentations.hit_fraction,'b',label='Lick Hit Fraction')
    ax[1].plot(session.stimulus_presentations.hit_rate,'r',label='Hit Rate')
    ax[1].plot(session.stimulus_presentations.false_alarm_rate,'k',label='False Alarm')
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].set_xlim([0,len(session.stimulus_presentations)])
    ax[1].set_ylim([0,1])
    ax[1].set_xlabel('Flash #',fontsize=16)
    ax[1].set_ylabel('Rate',fontsize=16)
    ax[1].tick_params(axis='both',labelsize=12)

    ax[2].plot(session.stimulus_presentations.d_prime,'k',label='d prime')
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[2].set_xlim([0,len(session.stimulus_presentations)])
    ax[2].set_ylim(bottom=-1)
    ax[2].set_xlabel('Flash #',fontsize=16)
    ax[2].set_ylabel('d prime',fontsize=16)
    ax[2].tick_params(axis='both',labelsize=12)
    ax[2].axhline(0,linestyle='--',alpha=0.5,color='k')

    plt.tight_layout()   
    if type(filename) is not type(None):
        plt.savefig(filename+".png")
 
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
    
def get_time_in_epochs(session):
    '''
        Computes the duration, in seconds, of each epoch in this session
        Returns a tuple (low-lick\low-reward, high-lick\high-reward, high-lick\low-reward)
    '''
    x0 = np.sum(session.stimulus_presentations.flash_metrics_epochs == 0) 
    x1 = np.sum(session.stimulus_presentations.flash_metrics_epochs == 1)
    x2 = np.sum(session.stimulus_presentations.flash_metrics_epochs == 2) 
    times = np.array([x0,x1,x2])*0.75    
    return times

'''
    Functions below here are for population analysis
'''


def plot_all_times(times,count,all_times):
    plt.figure(figsize=(5,5))
    labels = ['low-lick\nlow-reward','high-lick\nhigh-reward','high-lick\nlow-reward']
    means = np.mean(all_times/np.sum(all_times,1)[:,None],0)*100
    sem = np.std(all_times/np.sum(all_times,1)[:,None],0)/np.sqrt(count)*100
    colors = sns.color_palette(n_colors=3)   
    colors = np.vstack([colors[1], colors[0],colors[2]])
    for i in range(0,3):
        plt.plot([i-.5,i+.5],[means[i],means[i]],'-',color=colors[i],linewidth=4)
        plt.plot([i,i], [means[i]-sem[i], means[i]+sem[i]], 'k-')
    plt.xticks([0,1,2],labels,rotation=90,fontsize=20)
    plt.ylabel('% of time in each epoch',fontsize=20)
    plt.ylim([0,100])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

def plot_all_epochs(all_epochs):
    plt.figure(figsize=(10,5))
    colors = sns.color_palette(n_colors=3)  
    colors = np.vstack([colors[1], colors[0],colors[2]])
    labels = ['low-lick, low-reward','high-lick, high-reward','high-lick, low-reward']
    count = np.shape(all_epochs)[0]
    for i in range(0,3):
        plt.plot(np.sum(all_epochs==i,0)/count*100,color=colors[i],label=labels[i]) ## hard coded bug
    
    plt.ylim([0,100])
    plt.xlim([0,4790])
    plt.legend()
    plt.ylabel('% of session in each epoch',fontsize=20)
    plt.xlabel('Flash #',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
   
def plot_all_rates(all_lick,all_reward):
    plt.figure(figsize=(10,5))
    colors = sns.color_palette("hls",2)
    labels=['Lick Rate', 'Reward Rate']
    plt.plot(np.nanmean(all_lick,0),color=colors[0], label=labels[0]) 
    plt.plot(np.nanmean(all_reward,0),color=colors[1], label=labels[1]) 

    plt.ylim([0,0.25])
    plt.xlim([0,4790])
    plt.legend()
    plt.ylabel('Rate/Flash',fontsize=20)
    plt.xlabel('Flash #',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

def plot_all_dprime(all_dprime):
    plt.figure(figsize=(10,5))
    colors = sns.color_palette("hls",1)
    labels=['d prime']
    plt.plot(np.nanmean(all_dprime,0),color=colors[0], label=labels[0]) 

    plt.ylim([0,3])
    plt.xlim([0,4790])
    plt.legend()
    plt.ylabel('d prime',fontsize=20)
    plt.xlabel('Flash #',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()


def plot_all_performance_rates(all_hit_fraction,all_hit_rate, all_fa_rate):
    plt.figure(figsize=(10,5))
    colors = sns.color_palette("hls",3)
    labels=['Lick Hit Fraction', 'Hit Rate','False Alarm Rate']
    plt.plot(np.nanmean(all_hit_fraction,0),color=colors[0], label=labels[0]) 
    plt.plot(np.nanmean(all_hit_rate,0),color=colors[1], label=labels[1]) 
    plt.plot(np.nanmean(all_fa_rate,0),color=colors[2], label=labels[2]) 
    plt.ylim([0,1])
    plt.xlim([0,4790])
    plt.legend()
    plt.ylabel('Rate',fontsize=20)
    plt.xlabel('Flash #',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()




def compare_all_rates(all_lick,all_reward,rlabels):
    plt.figure(figsize=(10,5))

    labels=['Lick Rate', 'Reward Rate']
    if len(all_lick) == 2:  
        colors = sns.color_palette("hls",2)
        plt.plot(np.nanmean(all_lick[0],0),'-',color=colors[0], label=labels[0]+" " + rlabels[0]) 
        plt.plot(np.nanmean(all_reward[0],0),'-',color=colors[1], label=labels[1]+" " + rlabels[0]) 
        plt.plot(np.nanmean(all_lick[1],0),'--',color=colors[0], label=labels[0]+" " + rlabels[1]) 
        plt.plot(np.nanmean(all_reward[1],0),'--',color=colors[1], label=labels[1]+" " + rlabels[1])
        pvals = []
        for i in range(0,4790): 
            temp = ss.ttest_ind(all_lick[0][:,i],all_lick[1][:,i])
            pvals.append(temp.pvalue)
            if temp.pvalue < 0.05:
                plt.plot(i,0.001,'ks')
        pvals = np.array(pvals)
    else:
        colors = sns.color_palette("hls",len(all_lick))
        for i in range(0,len(all_lick)):
            plt.plot(np.nanmean(all_lick[i],0),'-',color=colors[i], label=labels[0]+" " + rlabels[i]) 
            plt.plot(np.nanmean(all_reward[i],0),'--',color=colors[i], label=labels[1]+" " + rlabels[i]) 
    plt.ylim([0,0.25])
    plt.xlim([0,4790])
    plt.legend()
    plt.ylabel('Rate/Flash')
    plt.xlabel('Flash #')
    plt.tight_layout()

def compare_all_performance_rates(all_hit_fraction,all_hit_rate,all_fa_rate,rlabels):
    plt.figure(figsize=(10,5))

    labels=['Lick Hit Fraction','Hit Rate','False Alarm Rate']
    if len(all_hit_fraction) == 2:  
        colors = sns.color_palette("hls",2)
        for i in range(0,len(all_hit_fraction)):
            plt.plot(np.nanmean(all_hit_fraction[i],0),'-',color=colors[i], label=labels[0]+" " + rlabels[i]) 
            plt.plot(np.nanmean(all_hit_rate[i],0),'--',color=colors[i], label=labels[1]+" " + rlabels[i]) 
            plt.plot(np.nanmean(all_fa_rate[i],0),'-.',color=colors[i], label=labels[1]+" " + rlabels[i]) 
        pvals = []
        for i in range(0,4790): 
            temp = ss.ttest_ind(all_hit_fraction[0][:,i],all_hit_fraction[1][:,i])
            pvals.append(temp.pvalue)
            if temp.pvalue < 0.05:
                plt.plot(i,0.001,'ks')
        pvals = np.array(pvals)
        pvals = []
        for i in range(0,4790): 
            temp = ss.ttest_ind(all_hit_rate[0][:,i],all_hit_rate[1][:,i])
            pvals.append(temp.pvalue)
            if temp.pvalue < 0.05:
                plt.plot(i,0.02,'ms')
        pvals = np.array(pvals)

    else:
        colors = sns.color_palette("hls",len(all_hit_fraction))
        for i in range(0,len(all_hit_fraction)):
            plt.plot(np.nanmean(all_hit_fraction[i],0),'-',color=colors[i], label=labels[0]+" " + rlabels[i]) 
            plt.plot(np.nanmean(all_hit_rate[i],0),'--',color=colors[i], label=labels[1]+" " + rlabels[i]) 
            plt.plot(np.nanmean(all_fa_rate[i],0),'-.',color=colors[i], label=labels[1]+" " + rlabels[i]) 
    plt.ylim([0,1])
    plt.xlim([0,4790])
    plt.legend()
    plt.ylabel('Rate')
    plt.xlabel('Flash #')
    plt.tight_layout()



def compare_all_dprime(all_dprime,rlabels):
    plt.figure(figsize=(10,5))

    labels=['dprime']
    if len(all_dprime) == 2:  
        colors = sns.color_palette("hls",4)
        plt.plot(np.nanmean(all_dprime[0],0),'-',color=colors[0], label=labels[0]+" " + rlabels[0]) 
        plt.plot(np.nanmean(all_dprime[1],0),'--',color=colors[0], label=labels[0]+" " + rlabels[1]) 

        pvals = []
        for i in range(0,4790): 
            temp = ss.ttest_ind(all_dprime[0][:,i],all_dprime[1][:,i])
            pvals.append(temp.pvalue)
            if temp.pvalue < 0.05:
                plt.plot(i,0.001,'ks')
        pvals = np.array(pvals)
    else:
        colors = sns.color_palette("hls",len(all_dprime))
        for i in range(0,len(all_dprime)):
            plt.plot(np.nanmean(all_dprime[i],0),'-',color=colors[i], label=labels[0]+" " + rlabels[i]) 
    plt.ylim(bottom=0)
    plt.xlim([0,4790])
    plt.legend()
    plt.ylabel('dprime')
    plt.xlabel('Flash #')
    plt.tight_layout()


def compare_all_epochs(all_epochs,rlabels,smoothing=0):
    plt.figure(figsize=(10,5))
    colors = sns.color_palette(n_colors=3)   
    labels = ['low-lick, low-reward','high-lick, high-reward','high-lick, low-reward']
    markers=['-','--','-o','-x']
    for j in range(0,len(all_epochs)):
        count = np.shape(all_epochs[j])[0]
        for i in range(0,3):
            if smoothing > 0:
               plt.plot(mov_avg(np.sum(all_epochs[j]==i,0)/count*100,n=smoothing),markers[j],color=colors[i],label=labels[i]+" "+rlabels[j])        
            else:
               plt.plot(np.sum(all_epochs[j]==i,0)/count*100,markers[j],color=colors[i],label=labels[i]+" "+rlabels[j])    
    plt.ylim([0,100])
    plt.xlim([0,4790])
    plt.legend()
    plt.ylabel('% of session in each epoch')
    plt.xlabel('Flash #')
    plt.tight_layout()

def plot_all_rates_averages(all_lick,all_reward):
    plt.figure(figsize=(5,5))
    labels = ['Lick Rate','Reward Rate']
    means = [np.nanmean(all_lick), np.nanmean(all_reward)]
    sem = [np.nanstd(all_lick)/np.sqrt(np.shape(all_lick)[0]), np.nanstd(all_reward)/np.sqrt(np.shape(all_lick)[0])]
    
    colors = sns.color_palette("hls",2)   
    for i in range(0,2):
        plt.plot([i-.5,i+.5],[means[i],means[i]],'-',color=colors[i],linewidth=4)
        plt.plot([i,i], [means[i]-sem[i], means[i]+sem[i]], 'k-')
    plt.xticks([0,1],labels,fontsize=16)
    plt.ylabel('Avg Rate/Flash',fontsize=20)
    plt.ylim([0,.25])
    plt.yticks(fontsize=16)
    plt.tight_layout()

def plot_all_performance_rates_averages(all_dprime,all_hit_fraction,all_hit_rate,all_fa_rate):
    plt.figure(figsize=(5,5))
    labels = ['dprime','Lick Hit Fraction','Hit Rate','False Alarm Rate']
    means = [np.nanmean(all_dprime), np.nanmean(all_hit_fraction), np.nanmean(all_hit_rate), np.nanmean(all_fa_rate)]
    sem = [np.nanstd(all_dprime)/np.sqrt(np.shape(all_dprime)[0]), np.nanstd(all_hit_fraction)/np.sqrt(np.shape(all_hit_fraction)[0]), np.nanstd(all_hit_rate)/np.sqrt(np.shape(all_hit_rate)[0]), np.nanstd(all_fa_rate)/np.sqrt(np.shape(all_fa_rate)[0])]
   
    colors = sns.color_palette("hls",4)   
    for i in range(0,4):
        plt.plot([i-.5,i+.5],[means[i],means[i]],'-',color=colors[i],linewidth=4)
        plt.plot([i,i], [means[i]-sem[i], means[i]+sem[i]], 'k-')
    plt.xticks([0,1,2,3],labels,fontsize=16,rotation=90)
    plt.ylabel('Avg Rate',fontsize=20)
    plt.ylim(bottom=0)
    plt.yticks(fontsize=16)
    plt.tight_layout()

def compare_hit_count(num_hitsA,num_hitsB):
    means = [np.mean(num_hitsA), np.mean(num_hitsB)] 
    sems = [stats.sem(num_hitsA), stats.sem(num_hitsB)]
    
    plt.figure(figsize=(5,5))
    colors = sns.color_palette("hls",2)
    w=0.4

    for j in range(0,len(means)):   
        plt.plot([j-w,j+w],[means[j],means[j]],'-',color=colors[j],linewidth=4)
        plt.plot([j,j], [means[j]-sems[j], means[j]+sems[j]], 'k-')
    
    plt.xticks([0,1],['A','B'],fontsize=12)
    plt.ylabel('Num Hits',fontsize=12)
    plt.ylim(0,150)
    plt.xlim(-0.5,1.5)
    plt.tight_layout()

def compare_all_performance_rates_averages_dprime(all_dprime,rlabels,split_on=None):
    plt.figure(figsize=(5,5))
    labels = ['']
    means=[]
    sems =[]
    maxnum=1
    diffsA = np.nanmean(all_dprime[0][:,0:split_on],1) - np.nanmean(all_dprime[0][:,split_on:],1)
    diffsB = np.nanmean(all_dprime[1][:,0:split_on],1) - np.nanmean(all_dprime[1][:,split_on:],1)

    for i in range(0,len(all_dprime)):    
        if not (type(split_on) == type(None)):
            labels = ['1st half','2nd half'] 
            maxnum=2
            means.append([np.nanmean(all_dprime[i][:,0:split_on]),          np.nanmean(all_dprime[i][:,split_on:]) ])
            sems.append([np.nanstd(all_dprime[i][:,0:split_on])/np.sqrt(np.shape(all_dprime[i][:,0:split_on])[0]), np.nanstd(all_dprime[i][:,split_on:])/np.sqrt(np.shape(all_dprime[i][:,split_on:])[0])])
        else: 
            means.append([np.nanmean(all_dprime[i])])
            sems.append([np.nanstd(all_dprime[i])/np.sqrt(np.shape(all_dprime[i])[0])])

    colors = sns.color_palette("hls",2)
    w = (1/len(all_dprime))/2- .05
    jw = 1/len(all_dprime)
    ldex = []
    lstr = []
    if maxnum == 2:
        colors = np.concatenate([colors, colors])
        
    for j in range(0,len(means)):   
        for i in range(0,maxnum):
            plt.plot([i+jw*j-w,i+jw*j+w],[means[j][i],means[j][i]],'-',color=colors[j],linewidth=4)
            plt.plot([i+jw*j,i+jw*j], [means[j][i]-sems[j][i], means[j][i]+sems[j][i]], 'k-')
            ldex.append(i+jw*j)
            lstr.append(labels[i]+" "+rlabels[j])



    if maxnum ==2:
        ylim = 2.5
        plt.plot([0.25,1.25],[ylim*1.05, ylim*1.05],'k-')
        plt.plot([0.25,0.25],[ylim, ylim*1.05], 'k-')
        plt.plot([1.25,1.25],[ylim, ylim*1.05], 'k-')
        if stats.ttest_ind(diffsA,diffsB)[1] < 0.05:
            plt.plot(.75, ylim*1.1,'k*')
        else:
            plt.text(.75,ylim*1.1, 'ns')


    plt.xticks(ldex,lstr,fontsize=12)
    plt.ylabel('dprime',fontsize=12)
    plt.ylim(0,3)
    plt.tight_layout() 





def compare_all_performance_rates_averages(all_dprime,all_hit_fraction,all_hit_rate,all_fa_rate,rlabels,split_on=None,color_alt=False):
    plt.figure(figsize=(5,5))
    labels = ['dprime','Lick Hit Fraction','Hit Rate','False Alarm Rate']
    means=[]
    sems =[]
    maxnum=4
    for i in range(0,len(all_dprime)):    
        if not (type(split_on) == type(None)):
            labels = ['dprime 1st','dprime 2nd','Lick Hit Fraction 1st','Lick Hit Fraction 2nd','Hit Rate 1st','Hit Rate 2nd', 'False Alarm Rate 1st', 'False Alarm Rate 2nd'] 
            maxnum=8
            means.append([
np.nanmean(all_dprime[i][:,0:split_on]),          np.nanmean(all_dprime[i][:,split_on:]), 
np.nanmean(all_hit_fraction[i][:,0:split_on]),    np.nanmean(all_hit_fraction[i][:,split_on:]), 
np.nanmean(all_hit_rate[i][:,0:split_on]),        np.nanmean(all_hit_rate[i][:,split_on:]), 
np.nanmean(all_fa_rate[i][:,0:split_on]),         np.nanmean(all_fa_rate[i][:,split_on:])
])
            sems.append([
np.nanstd(all_dprime[i][:,0:split_on])/np.sqrt(np.shape(all_dprime[i][:,0:split_on])[0]), np.nanstd(all_dprime[i][:,split_on:])/np.sqrt(np.shape(all_dprime[i][:,split_on:])[0]), 
np.nanstd(all_hit_fraction[i][:,0:split_on])/np.sqrt(np.shape(all_hit_fraction[i][:,0:split_on])[0]), np.nanstd(all_hit_fraction[i][:,split_on:])/np.sqrt(np.shape(all_hit_fraction[i][:,split_on:])[0]), 
np.nanstd(all_hit_rate[i][:,0:split_on])/np.sqrt(np.shape(all_hit_rate[i][:,0:split_on])[0]), np.nanstd(all_hit_rate[i][:,split_on:])/np.sqrt(np.shape(all_hit_rate[i][:,split_on:])[0]), 
np.nanstd(all_fa_rate[i][:,0:split_on])/np.sqrt(np.shape(all_fa_rate[i][:,0:split_on])[0]), np.nanstd(all_fa_rate[i][:,split_on:])/np.sqrt(np.shape(all_fa_rate[i][:,split_on:])[0])
])
        else: 
            means.append([np.nanmean(all_dprime[i]), np.nanmean(all_hit_fraction[i]), np.nanmean(all_hit_rate[i]), np.nanmean(all_fa_rate[i])])
            sems.append([np.nanstd(all_dprime[i])/np.sqrt(np.shape(all_dprime[i])[0]), np.nanstd(all_hit_fraction[i])/np.sqrt(np.shape(all_hit_fraction[i])[0]), np.nanstd(all_hit_rate[i])/np.sqrt(np.shape(all_hit_rate[i])[0]), np.nanstd(all_fa_rate[i])/np.sqrt(np.shape(all_fa_rate[i])[0])])

    colors = sns.color_palette("hls",4)
    w = (1/len(all_dprime))/2- .05
    jw = 1/len(all_dprime)
    ldex = []
    lstr = []
    if maxnum == 8:
        colors = np.repeat(np.vstack(colors),2,axis=0)
        
    for j in range(0,len(means)):   
        for i in range(0,maxnum):
            if color_alt:
                plt.plot([i+jw*j-w,i+jw*j+w],[means[j][i],means[j][i]],'-',color=colors[j],linewidth=4)
            else:
                plt.plot([i+jw*j-w,i+jw*j+w],[means[j][i],means[j][i]],'-',color=colors[i],linewidth=4)
            plt.plot([i+jw*j,i+jw*j], [means[j][i]-sems[j][i], means[j][i]+sems[j][i]], 'k-')
            ldex.append(i+jw*j)
            lstr.append(labels[i]+" "+rlabels[j])

    plt.xticks(ldex,lstr,rotation=90)
    plt.ylabel('Avg Rate')
    plt.ylim(bottom=0)
    plt.tight_layout() 

def compare_all_rates_averages(all_lick,all_reward,rlabels,split_on=None):
    plt.figure(figsize=(5,5))
    labels = ['Lick Rate','Reward Rate']
    means=[]
    sems =[]
    maxnum=2
    for i in range(0,len(all_lick)):
        if not (type(split_on) == type(None)):
            labels =  ['Lick Rate 1st','Lick Rate 2nd','Reward Rate 1st', 'Reward Rate 2nd']
            maxnum=4
            means.append([
np.nanmean(all_lick[i][:,0:split_on]),np.nanmean(all_lick[i][:,split_on:]), 
np.nanmean(all_reward[i][:,0:split_on]), np.nanmean(all_reward[i][:,split_on:])])
            sems.append([
np.nanstd(all_lick[i][:,0:split_on])/np.sqrt(np.shape(all_lick[i][:,0:split_on])[0]), np.nanstd(all_lick[i][:,split_on:])/np.sqrt(np.shape(all_lick[i][:,split_on:])[0]), 
np.nanstd(all_reward[i][:,0:split_on])/np.sqrt(np.shape(all_lick[i][:,0:split_on])[0]), np.nanstd(all_reward[i][:,split_on:])/np.sqrt(np.shape(all_lick[i][:,split_on:])[0])
])
        else:
            means.append([np.nanmean(all_lick[i]), np.nanmean(all_reward[i])])
            sems.append([np.nanstd(all_lick[i])/np.sqrt(np.shape(all_lick[i])[0]), np.nanstd(all_reward[i])/np.sqrt(np.shape(all_lick[i])[0])])
    
    colors = sns.color_palette("hls",2)
    if maxnum == 4:
        colors = np.repeat(np.vstack(colors),2,axis=0)
    w = (1/len(all_lick))/2- .05
    jw = 1/len(all_lick)
    ldex = []
    lstr = []
    for j in range(0,len(means)):   
        for i in range(0,maxnum):
            plt.plot([i+jw*j-w,i+jw*j+w],[means[j][i],means[j][i]],'-',color=colors[i],linewidth=4)
            plt.plot([i+jw*j,i+jw*j], [means[j][i]-sems[j][i], means[j][i]+sems[j][i]], 'k-')
            ldex.append(i+jw*j)
            lstr.append(labels[i]+" "+rlabels[j])

    plt.xticks(ldex,lstr,rotation=90)
    plt.ylabel('Avg Rate/Flash')
    plt.ylim([0,.25])
    plt.tight_layout()

def compare_all_times(times,count,all_times,rlabels):
    plt.figure(figsize=(5,5))
    labels = ['low-lick\nlow-reward','high-lick\nhigh-reward','high-lick\nlow-reward']
    means=[]
    sems = []
    for i in range(0,len(times)):
        means.append(np.mean(all_times[i]/np.sum(all_times[i],1)[:,None],0)*100)
        sems.append(np.std(all_times[i]/np.sum(all_times[i],1)[:,None],0)/np.sqrt(count[i])*100)
    colors = sns.color_palette(n_colors=3)  
    w = (1/len(all_times))/2- .05
    jw = 1/len(all_times)
    ldex = []
    lstr = []
    for j in range(0,len(all_times)):
        for i in range(0,3):
            plt.plot([i+jw*j-w,i+jw*j+w],[means[j][i],means[j][i]],'-',color=colors[i],linewidth=4)
            plt.plot([i+jw*j,i+jw*j], [means[j][i]-sems[j][i], means[j][i]+sems[j][i]], 'k-')
            ldex.append(i+jw*j)
            lstr.append(labels[i]+" "+rlabels[j])           
    plt.xticks(ldex,lstr,rotation=90)
    plt.ylabel('% of time in each epoch')
    plt.ylim([0,100])
    plt.tight_layout()

def get_rates(ids=None):
    '''
        Computes summary info for all sessions in ids
    '''
    if type(ids) == type(None):
        ids = pgt.get_active_ids()
    lick_rate = []
    reward_rate = []
    epochs = []
    hit_fraction =[]
    hit_rate = []
    fa_rate = []
    dprime=[]

    times = np.zeros(3,)
    count = 0
    all_times = []

    for id in ids:
        print(id)
        try:
            session = pgt.get_data(id)
            get_metrics(session)

            lick_rate.append(session.stimulus_presentations['bout_rate'].values)
            reward_rate.append(session.stimulus_presentations['reward_rate'].values)
            hit_fraction.append(session.stimulus_presentations['hit_fraction'].values)
            hit_rate.append(session.stimulus_presentations['hit_rate'].values)
            fa_rate.append(session.stimulus_presentations['false_alarm_rate'].values)
            dprime.append(session.stimulus_presentations['d_prime'].values)

            my_epochs = session.stimulus_presentations['flash_metrics_epochs'].values
            epochs.append(my_epochs)

            my_times = get_time_in_epochs(session)
            times += my_times
            count +=1
            all_times.append(my_times)

        except:
            print(' crash')
    
    lens = [len(x) for x in lick_rate]
    all_lick = np.zeros((len(lick_rate), np.max(lens)))
    all_lick[:] = np.nan
    all_reward = np.zeros((len(lick_rate), np.max(lens)))
    all_reward[:] = np.nan
    all_hit_fraction = np.zeros((len(lick_rate), np.max(lens)))
    all_hit_fraction[:] = np.nan
    all_hit_rate = np.zeros((len(lick_rate), np.max(lens)))
    all_hit_rate[:] = np.nan
    all_fa_rate = np.zeros((len(lick_rate), np.max(lens)))
    all_fa_rate[:] = np.nan
    all_dprime = np.zeros((len(lick_rate), np.max(lens)))
    all_dprime[:] = np.nan
    for i in range(0,len(lick_rate)):
        all_lick[i,0:len(lick_rate[i])] = lick_rate[i]   
        all_reward[i,0:len(reward_rate[i])] = reward_rate[i]   
        all_hit_fraction[i,0:len(hit_fraction[i])] = hit_fraction[i]   
        all_hit_rate[i,0:len(hit_rate[i])] = hit_rate[i]   
        all_fa_rate[i,0:len(fa_rate[i])] = fa_rate[i]   
        all_dprime[i,0:len(dprime[i])] = dprime[i]   

    lens = [len(x) for x in epochs]
    all_epochs = np.zeros((len(epochs), np.max(lens)))
    all_epochs[:] = np.nan
    for i in range(0,len(epochs)):
        all_epochs[i,0:len(epochs[i])] = epochs[i]

    all_times = np.vstack(all_times)
    return all_lick, all_reward,all_epochs, times, count,all_times, all_hit_fraction, all_hit_rate, all_fa_rate, all_dprime

def mov_avg(a,n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_num_hits(ids):
    num_hits = []
    for index, id in enumerate(ids):
        session = pgt.get_data(id)
        num_hits.append(np.sum(session.trials.hit)) 
    return num_hits 
