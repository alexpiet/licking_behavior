import psy_tools as ps
import psy_timing_tools as pt
from alex_utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as  sns
import pandas as pd
import matplotlib.patches as patches

def annotate_bouts(session):
    if 'bout_start' not in session.licks:
        pt.annotate_licks(session)
    bout_starts = session.licks[session.licks['bout_start']]
    session.stimulus_presentations['bout_start'] = False
    for index,x in bout_starts.iterrows():
        filter_start = session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)]
        if len(filter_start) > 0:
            session.stimulus_presentations.at[session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)].index[0]-1,'bout_start'] = True
    session.stimulus_presentations.drop(-1,inplace=True)

def annotate_flash_rolling_metrics(session,win_dur=320, win_type='triang'):
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
    if use_bouts:
        session.stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in session.stimulus_presentations['bout_rate']] 
    else:
        session.stimulus_presentations['high_lick'] = [True if x > lick_threshold else False for x in session.stimulus_presentations['lick_rate']] 
    session.stimulus_presentations['high_reward'] = [True if x > reward_threshold else False for x in session.stimulus_presentations['reward_rate']] 
    session.stimulus_presentations['flash_metrics_epochs'] = [0 if (not x[0]) & (not x[1]) else 1 if x[1] else 2 for x in zip(session.stimulus_presentations['high_lick'], session.stimulus_presentations['high_reward'])]
    session.stimulus_presentations['flash_metrics_labels'] = ['low-lick,low-reward' if x==0  else 'high-lick,high-reward' if x==1 else 'high-lick,low-reward' for x in session.stimulus_presentations['flash_metrics_epochs']]

def plot_metrics(session,use_bouts=True,filename=None):
    plt.figure(figsize=(11,4))
    if 'bout_rate' not in session.stimulus_presentations:
        annotate_flash_rolling_metrics(session)
        classify_by_flash_metrics(session)
    
    cluster_labels = session.stimulus_presentations['flash_metrics_epochs'].values
    cluster_colors = sns.color_palette("hls",3)
    cp = np.where(~(np.diff(cluster_labels) == 0))[0]
    cp = np.concatenate([[0], cp, [len(cluster_labels)]])
    plotted = np.zeros(3,)
    labels = ['low-lick,low-reward','high-lick,high-reward','high-lick,low-reward']
    for i in range(0, len(cp)-1):
        if plotted[cluster_labels[cp[i]+1]]:
            plt.gca().axvspan(cp[i],cp[i+1],color=cluster_colors[cluster_labels[cp[i]+1]], alpha=0.2)
        else:
            plotted[cluster_labels[cp[i]+1]] = True
            plt.gca().axvspan(cp[i],cp[i+1],color=cluster_colors[cluster_labels[cp[i]+1]], alpha=0.2,label=labels[cluster_labels[cp[i]+1]])

    plt.plot(session.stimulus_presentations.reward_rate,'m',label='Flash Reward')
    if use_bouts:
        plt.plot(session.stimulus_presentations.bout_rate,'g',label='Flash Lick')
    else:
        plt.plot(session.stimulus_presentations.lick_rate,'g',label='Flash Lick')
    plt.gca().axhline(0,linestyle='--',alpha=0.5,color='k')
    plt.gca().axhline(2/80,linestyle='--',alpha=0.5,color='m')
    plt.gca().axhline(.1,linestyle='--',alpha=0.5,color='g')
    plt.xlabel('Flash #',fontsize=12)
    plt.ylabel('Rate/Flash',fontsize=12)
    plt.legend()
    plt.xlim([0,len(session.stimulus_presentations)])
    plt.ylim([0,1])
    plt.tight_layout()   
    if type(filename) is not type(None):
        plt.savefig(filename+".png")
 
def plot_2D(session,lick_threshold = 0.1, reward_threshold = 2/80,filename=None):
    plt.figure()
    if 'bout_rate' not in session.stimulus_presentations:
        annotate_flash_rolling_metrics(session)  
    cluster_colors = sns.color_palette("hls",3)   
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
    
    






