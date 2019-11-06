import numpy as np
import matplotlib.pyplot as plt
import psy_general_tools as pgt
import seaborn as sns
import pandas as pd
import matplotlib.patches as patches
import scipy.stats as ss

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
    for index,x in bout_starts.iterrows():
        filter_start = session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)]
        if len(filter_start) > 0:
            session.stimulus_presentations.at[session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)].index[0]-1,'bout_start'] = True

    # Annotate Bout Ends
    bout_ends = session.licks[session.licks['bout_end']]
    session.stimulus_presentations['bout_end'] = False
    for index,x in bout_ends.iterrows():
        filter_start = session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)]
        if len(filter_start) > 0:
            session.stimulus_presentations.at[session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)].index[0]-1,'bout_end'] = True

    # Clean Up
    session.stimulus_presentations.drop(-1,inplace=True,errors='ignore')

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
    plt.figure(figsize=(10,5))
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
            plt.gca().axvspan(cp[i],cp[i+1],color=cluster_colors[cluster_labels[cp[i]+1]], alpha=0.2)
        else:
            plotted[cluster_labels[cp[i]+1]] = True
            plt.gca().axvspan(cp[i],cp[i+1],color=cluster_colors[cluster_labels[cp[i]+1]], alpha=0.2,label=labels[cluster_labels[cp[i]+1]])

    plt.plot(session.stimulus_presentations.reward_rate,'m',label='Reward Rate')
    plt.gca().axhline(2/80,linestyle='--',alpha=0.5,color='m',label='Reward Threshold')
    if use_bouts:
        plt.plot(session.stimulus_presentations.bout_rate,'g',label='Lick Rate')
    else:
        plt.plot(session.stimulus_presentations.lick_rate,'g',label='Flash Lick')
    plt.gca().axhline(.1,linestyle='--',alpha=0.5,color='g',label='Lick Threshold')
    plt.xlabel('Flash #',fontsize=20)
    plt.ylabel('Rate/Flash',fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.xlim([0,len(session.stimulus_presentations)])
    plt.ylim([0,1])
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

def compare_all_rates_averages(all_lick,all_reward,rlabels):
    plt.figure(figsize=(5,5))
    labels = ['Lick Rate','Reward Rate']
    means=[]
    sems =[]
    for i in range(0,len(all_lick)):
        means.append([np.nanmean(all_lick[i]), np.nanmean(all_reward[i])])
        sems.append([np.nanstd(all_lick[i])/np.sqrt(np.shape(all_lick[i])[0]), np.nanstd(all_reward[i])/np.sqrt(np.shape(all_lick[i])[0])])
    
    colors = sns.color_palette("hls",2)
    w = (1/len(all_lick))/2- .05
    jw = 1/len(all_lick)
    ldex = []
    lstr = []
    for j in range(0,len(all_lick)):   
        for i in range(0,2):
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
    for i in range(0,len(lick_rate)):
        all_lick[i,0:len(lick_rate[i])] = lick_rate[i]   
        all_reward[i,0:len(reward_rate[i])] = reward_rate[i]   

    lens = [len(x) for x in epochs]
    all_epochs = np.zeros((len(epochs), np.max(lens)))
    all_epochs[:] = np.nan
    for i in range(0,len(epochs)):
        all_epochs[i,0:len(epochs[i])] = epochs[i]

    all_times = np.vstack(all_times)
    return all_lick, all_reward,all_epochs, times, count,all_times

def mov_avg(a,n=5):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
