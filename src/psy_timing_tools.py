import psy_general_tools as pgt
import matplotlib.pyplot as plt
import psy_metrics_tools as pm
import numpy as np
import seaborn
import pandas as pd
import matplotlib.patches as patches
from tqdm import tqdm
import psy_style as pstyle


# TODO, Issue #233
def plot_bout_ibi(bout,from_start=False,directory=None):
    plt.figure()
    if from_start:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ibi'] < 10)]['post_ibi_from_start'], bins = 25,color='k',label='post-miss')
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ibi'] < 10)]['post_ibi_from_start'],bins=h[1], color='r',label='post-reward')
    else:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ibi'] < 10)]['post_ibi'], bins = 25,color='k',label='post-miss')
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ibi'] < 10)]['post_ibi'],bins=h[1], color='r',label='post-reward')
    plt.legend()
    plt.ylabel('count',fontsize=12)
    if from_start:
        plt.xlabel('InterLick (s), time from bout start', fontsize=12)
    else:
        plt.xlabel('InterLick (s), time from bout end', fontsize=12)
    ylims = plt.ylim()
    mean_all, mean_reward = get_bout_ibi(bout, from_start=from_start)
    plt.plot(mean_all, ylims[1], 'kv')    
    plt.plot(mean_reward, ylims[1], 'rv') 
    #if type(directory) is not type(None):
    #    if from_start:
    #        plt.savefig(directory+"bout_ili_distribution_from_start.svg")
    #    else:
    #        plt.savefig(directory+"bout_ili_distribution_from_end.svg")

# TODO, Issue #233
def plot_bout_ibi_current(bout,from_start=False,directory=None):
    plt.figure()
    if from_start:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ibi'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ibi_from_start'], bins = 25,color='b',label='post-miss miss',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ibi'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ibi_from_start'],bins=h[1], color='m',label='post-reward miss',alpha=.25)
        plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ibi'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ibi_from_start'],bins=h[1], color='k',label='post-miss hit',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ibi'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ibi_from_start'],bins=h[1], color='r',label='post-reward hit',alpha=.25)
    else:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ibi'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ibi'], bins = 25,color='b',label='post-miss miss',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ibi'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ibi'],bins=h[1], color='m',label='post-reward miss',alpha=.25)
        plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ibi'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ibi'],bins=h[1], color='k',label='post-miss hit',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ibi'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ibi'],bins=h[1], color='r',label='post-reward hit',alpha=.25)
    plt.legend()
    plt.ylabel('count',fontsize=12)
    if from_start:
        plt.xlabel('InterLick (s), time from bout start', fontsize=12)
    else:
        plt.xlabel('InterLick (s), time from bout end', fontsize=12)
    ylims = plt.ylim()
    mean_miss, mean_reward = get_bout_ibi_current(bout, from_start=from_start,current_hit=True)
    plt.plot(mean_miss, ylims[1], 'kv')    
    plt.plot(mean_reward, ylims[1], 'rv') 
    mean_miss, mean_reward = get_bout_ibi_current(bout, from_start=from_start,current_hit=False)
    plt.plot(mean_miss, ylims[1], 'b^')    
    plt.plot(mean_reward, ylims[1], 'm^') 
    #if type(directory) is not type(None):
    #    if from_start:
    #        plt.savefig(directory+"bout_ili_distribution_from_start_current_hitmiss.svg")
    #    else:
    #        plt.savefig(directory+"bout_ili_distribution_from_end_current_hitmiss.svg")

 # TODO, Issue #233
def get_bout_ibi(bout, from_start=False):
    if from_start:
        mean_miss = np.nanmean(bout[(~bout['bout_rewarded']) & (bout['post_ibi'] < 10)]['post_ibi_from_start'])
        mean_reward = np.nanmean(bout[(bout['bout_rewarded']) & (bout['post_ibi'] < 10)]['post_ibi_from_start'])
    else:
        mean_miss = np.nanmean(bout[(~bout['bout_rewarded']) & (bout['post_ibi'] < 10)]['post_ibi'])
        mean_reward = np.nanmean(bout[(bout['bout_rewarded']) & (bout['post_ibi'] < 10)]['post_ibi'])
    return mean_miss, mean_reward

# TODO, Issue #233
def get_bout_ibi_current(bout,from_start=False, current_hit=True):
    if from_start:
        start_str = 'post_ibi_from_start'
    else:
        start_str = 'post_ibi'
    if current_hit:
        mean_miss = np.nanmean( bout[(~bout['bout_rewarded'])&(bout['post_ibi']<10)&(bout['bout_rewarded'].shift(-1))][start_str])
        mean_reward = np.nanmean( bout[(bout['bout_rewarded'])&(bout['post_ibi']<10)&(bout['bout_rewarded'].shift(-1))][start_str])
    else:
        mean_miss = np.nanmean( bout[(~bout['bout_rewarded'])&(bout['post_ibi']<10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))][start_str])
        mean_reward = np.nanmean( bout[(bout['bout_rewarded'])&(bout['post_ibi']<10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))][start_str])  
    return mean_miss, mean_reward

# TODO, Issue #233
def plot_all_bout_statistics(durs,all_bout=None,directory=None):
    # the mean post miss is 4 flashes + 2.3 flasshes
    # 2.3 =mean(geometric(.3)) = 1/.3
    # each flash is 0.75 seconds, and you want to respond anywhere in that range so:
    optimal_post_miss = [5.3*.75+.3, 6.3*.75+.3+.75]  
    # Post hit,  is 3.75 to 4.5 seconds of grace period, then the optimal post_miss
    optimal_post_hit = [3.75 + 5.3*.75, 4.5 + 6.3*.75]
    durs = np.vstack(durs)
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(durs[:,2], durs[:,3], 'ko',alpha=.2)
    ax[0].plot([0,10],[0,10],'k--',alpha=.2)
    ax[0].set_ylabel('mean IBI (s), post-hit \n from start of bout',fontsize=12)
    ax[0].set_xlabel('mean IBI (s), post-miss \n from start of bout',fontsize=12)
    ax[1].plot(durs[:,0], durs[:,1], 'ko',alpha=.2)
    ax[1].plot([0,10],[0,10],'k--',alpha=.2)
    ax[1].set_ylabel('mean IBI (s), post-hit \n from end of bout',fontsize=12)
    ax[1].set_xlabel('mean IBI (s), post-miss \n from end of bout',fontsize=12)
    plt.tight_layout()
    if type(all_bout) is not type(None):
        from_start_durs = get_bout_ili(all_bout, from_start=True)
        ax[0].plot(from_start_durs[0], from_start_durs[1],'ko')
        from_end_durs = get_bout_ili(all_bout, from_start=False)
        ax[1].plot(from_end_durs[0], from_end_durs[1],'ko')
    x = optimal_post_miss[0]
    xsize = np.diff(optimal_post_miss)
    y = optimal_post_hit[0]
    ysize = np.diff(optimal_post_hit)
    rect = patches.Rectangle((x,y),xsize,ysize,linewidth=1, edgecolor='m', facecolor='m',alpha=.25)
    ax[0].add_patch(rect)
    if type(directory) is not type(None):
        plt.savefig(directory+"bout_ili_scatter.svg")

# TODO, Issue #233
def plot_all_bout_statistics_current(durs,all_bout=None,directory=None):
    optimal_post_miss = [5.3*.75+.3, 6.3*.75+.3+.75]  
    optimal_post_hit = [3.75 + 5.3*.75, 4.5 + 6.3*.75]
    durs = np.vstack(durs)
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(durs[:,6], durs[:,7], 'bo',alpha=.2)
    ax[0].plot(durs[:,10], durs[:,11], 'ro',alpha=.2)
    ax[0].plot([0,10],[0,10],'k--',alpha=.2)
    ax[0].set_ylabel('mean IBI (s) post-hit \n from start of bout',fontsize=12)
    ax[0].set_xlabel('mean IBI (s) post-miss \n from start of bout',fontsize=12)
    ax[1].plot(durs[:,4], durs[:,5], 'bo',alpha=.2)
    ax[1].plot(durs[:,8], durs[:,9], 'ro',alpha=.2)
    ax[1].plot([0,10],[0,10],'k--',alpha=.2)
    ax[1].set_ylabel('mean IBI (s) post-hit \n from end of bout',fontsize=12)
    ax[1].set_xlabel('mean IBI (s) post-miss \n from end of bout',fontsize=12)
    plt.tight_layout()
    if type(all_bout) is not type(None):
        from_start_durs = get_bout_ili_current(all_bout, from_start=True,current_hit=True)
        ax[0].plot(from_start_durs[0], from_start_durs[1],'ro',label='Current Hit')
        from_end_durs = get_bout_ili_current(all_bout, from_start=False,current_hit=True)
        ax[1].plot(from_end_durs[0], from_end_durs[1],'ro')

        from_start_durs_miss = get_bout_ili_current(all_bout, from_start=True,current_hit=False)
        ax[0].plot(from_start_durs_miss[0], from_start_durs_miss[1],'bo',label='Current Miss')
        from_end_durs_miss = get_bout_ili_current(all_bout, from_start=False,current_hit=False)
        ax[1].plot(from_end_durs_miss[0], from_end_durs_miss[1],'bo')
    x = optimal_post_miss[0]
    xsize = np.diff(optimal_post_miss)
    y = optimal_post_hit[0]
    ysize = np.diff(optimal_post_hit)
    rect = patches.Rectangle((x,y),xsize,ysize,linewidth=1, edgecolor='m', facecolor='m',alpha=.25)
    ax[0].add_patch(rect)
    ax[0].legend()    
    if type(directory) is not type(None):
        plt.savefig(directory+"bout_ili_scatter_by_current_hitmiss.svg")
