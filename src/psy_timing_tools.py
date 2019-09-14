import psy_tools as ps
from alex_utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import matplotlib.patches as patches


def plot_all_mouse_durations(all_durs):
    plt.figure()
    for dur in all_durs:
        if len(dur) == 4:
            plt.plot(dur,'o-')
    plt.ylabel('Avg ILI (s)')
    plt.xlabel('Ophys Session #')
    plt.ylim(bottom=0)

def get_all_mouse_durations(mice_ids):
    all_durs=[]
    for mouse in mice_ids:
        print(str(mouse))
        durs = get_mouse_durations(mouse)
        all_durs.append(durs)
    return all_durs

def get_mouse_durations(mouse_id):
    sessions,IDS,active = ps.load_mouse(mouse_id)
    durs = []
    for sess in np.array(sessions)[active]:
        durs.append(get_mean_lick_distribution(sess))
    return durs


def get_lick_count(id):
    session = ps.get_data(id)
    annotate_licks(session)
    d = session.licks['pre_ili'][session.licks.rewarded]
    hits = len(d[(d>.7)& (d<10)].values)
    d = session.licks['pre_ili']
    total = len(d[(d>.7)& (d<10)].values)   
    return total, hits

def annotate_licks(session,bout_threshold=0.7):
    if 'bout_number' in session.licks:
        raise Exception('You already annotated this session, reload session first')
    licks = session.licks
    licks['pre_ili'] = np.concatenate([[np.nan],np.diff(licks.timestamps.values)])
    licks['post_ili'] = np.concatenate([np.diff(licks.timestamps.values),[np.nan]])
    licks['rewarded'] = False
    for index, row in session.rewards.iterrows():
        mylick = np.where(licks.timestamps <= row.timestamps)[0][-1]
        licks.at[mylick,'rewarded'] = True
    licks['bout_start'] = licks['pre_ili'] > bout_threshold
    licks['bout_end'] = licks['post_ili'] > bout_threshold
    licks.at[licks['pre_ili'].apply(np.isnan),'bout_start']=True
    licks.at[licks['post_ili'].apply(np.isnan),'bout_end']=True
    licks['bout_number'] = np.cumsum(licks['bout_start'])
    x = session.licks.groupby('bout_number').any('rewarded').rename(columns={'rewarded':'bout_rewarded'})
    session.licks['bout_rewarded'] = False
    temp = session.licks.reset_index().set_index('bout_number')
    temp.update(x)
    temp = temp.reset_index().set_index('index')
    session.licks['bout_rewarded'] = temp['bout_rewarded']

def plot_mouse_lick_distributions(id,nbins=50,directory=None):
    sessions, ids, active = ps.load_mouse(id)
    sessions = np.array(sessions)[np.array(active)]
    fig,ax = plt.subplots(2,2)
    if len(sessions) > 0:
        plot_mouse_lick_distributions_inner(sessions[0], ax[0,0],nbins,id)
    if len(sessions) > 1:
        plot_mouse_lick_distributions_inner(sessions[1], ax[0,1],nbins,id)
    if len(sessions) > 2:
        plot_mouse_lick_distributions_inner(sessions[2], ax[1,0],nbins,id)
    if len(sessions) > 3:
        plot_mouse_lick_distributions_inner(sessions[3], ax[1,1],nbins,id)
    plt.tight_layout()
    if type(directory) is not type(None):
        plt.savefig(directory+"mouse_"+str(id)+"_ILI.png")

def plot_mouse_lick_distributions_inner(session, ax,nbins,id):
    annotate_licks(session)
    licks = session.licks.timestamps.values
    diffs = np.diff(licks)
    h=ax.hist(diffs[diffs<10],nbins,label='All')
    ax.axvline(0.75,linestyle='--',color='k')
    ax.set_ylabel('count')
    ax.set_xlabel('InterLick (s)')
    ax.set_ylim([0,100])
    ax.set_title(str(id)+" "+session.metadata['stage'])
    m = get_mean_lick_distribution(session)
    ax.axvline(m,linestyle='--',color='r')
    d = session.licks['pre_ili'][session.licks.rewarded]
    h2= ax.hist(d[(d>.7)&(d<10)],bins=h[1],label='Hits')

# Make Figure of distribution of licks
def plot_lick_distribution(session,nbins=50,directory=None):
    #annotate_licks(session)
    licks = session.licks.timestamps.values
    diffs = np.diff(licks) 
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    h=ax[0].hist(diffs[diffs<10],nbins,label='All')
    ax[0].axvline(0.75,linestyle='--',color='k')
    ax[0].set_ylabel('count')
    ax[0].set_xlabel('InterLick (s)')
    ax[0].set_ylim([0,100])
    ax[0].set_title(str(session.metadata['mouse_id'])+" "+session.metadata['stage'])
    m = get_mean_lick_distribution(session)
    ax[0].axvline(m,linestyle='--',color='r')
    d = session.licks['pre_ili'][session.licks.rewarded]
    h2= ax[0].hist(d[(d>.7)&(d<10)],bins=h[1],label='Hits')
    ax[0].legend()
    h1 = np.histogram(diffs[(diffs>.7)&(diffs<10)],bins=h[1])
    centers = np.diff(h[1]) + h[1][0:-1]
    ax[1].plot(centers[centers > .7], (h1[0]-h2[0])[centers > .7]  ,'k-')
    ax[1].set_ylabel('Miss Licks')
    ax[1].set_xlabel('InterLick (s)')
    if type(directory) is not type(None):
        id = session.metadata['ophys_experiment_id']
        plt.savefig(directory+str(id)+"_ILI.png")

def get_chronometric(bout,nbins=50, filename=None,title = ''): 
    d = bout['pre_ili']
    dr = bout[bout['bout_rewarded']]['pre_ili']
    h= np.histogram(d[(d>.7)&(d<10)],nbins)
    h1= np.histogram(dr[(dr>.7)&(dr<10)],bins=h[1])
    centers = np.diff(h[1]) + h[1][0:-1]
    fig, ax = plt.subplots(2,1)
    chrono = np.array(h1[0])/h[0]
    ax[0].plot(centers[centers > .7], h[0][centers > .7], 'k',label='All')
    ax[0].plot(centers[centers > .7], h1[0][centers > .7], 'r',label='Hits')
    ax[0].set_ylabel('Count',fontsize=12)
    ax[0].set_xlabel('InterLick (s)',fontsize=12)
    ax[0].legend()
    c = centers[centers > .7]
    chron = chrono[centers > .7]
    err = 1.96*np.sqrt(chron*(1-chron)/h[0]) 
    ax[1].errorbar(c,chron,yerr = err,color='k', ecolor='lightgray',elinewidth=5,capsize=0)
    ax[1].set_ylabel('% Hit',fontsize=12)
    ax[1].set_xlabel('InterLick (s)',fontsize=12)
    ax[1].set_ylim([-0.025, .5])
    ax[0].set_title(title)
    plt.tight_layout()
    if type(filename) is not type(None):
        plt.savefig(filename+"_chronometric.png")

def plot_all_session_chronometric(IDS,nbins=15):
    for id in IDS:
        print(id)
        try:
            session = ps.get_data(id)
            if len(session.licks) > 10:
                annotate_licks(session) 
                bout = get_bout_table(session)
                filename = '/home/alex.piet/codebase/behavior/psy_fits_v2/' + str(id)
                get_chronometric(bout,nbins=nbins,filename=filename,title= 'Session ' + str(id))
        except:
            print(' crash')
    
def plot_all_mice_chronometric(IDS,nbins=25):
    for id in IDS:
        print(id)
        try:
            mice_ids = ps.get_mice_sessions(id)
            bout = get_all_bout_table(mice_ids)
            filename = '/home/alex.piet/codebase/behavior/psy_fits_v2/mouse_' + str(id)
            get_chronometric(bout,nbins=nbins,filename=filename,title='mouse ' + str(id))
        except:
            print(' crash')
    
def get_mean_lick_distribution(session):
    licks = session.licks.timestamps.values
    diffs = np.diff(licks)
    good_diffs = diffs[(diffs<10) & (diffs > 0.75)]
    return np.mean(good_diffs)


def plot_session(session):
    '''
    Evaluate fit by plotting prediction and lick times

    Args:
        Latent: a vector of the estimate lick rate
        time_vec: the timestamp for each time bin
        licks: the time of each lick in dt-rounded timevalues
        stop_time: the number of timebins
    
    Plots the lick raster, and latent rate
    
    Returns: the figure handle and axis handle
    '''
    colors = seaborn.color_palette('hls',8)
    fig,axes  = plt.subplots()  
    fig.set_size_inches(12,4) 
    axes.set_ylim([0, 1])
    axes.set_xlim(600,650)
    tt= .7
    bb = .3
    for index, row in session.stimulus_presentations.iterrows():
        if not row.omitted:
            axes.axvspan(row.start_time,row.stop_time, alpha=0.2,color='k', label='flash')
        if row.change:
            axes.axvspan(row.start_time,row.stop_time, alpha=0.6,color='c', label='change flash')
    bouts = session.licks.bout_number.unique()
    for b in bouts:
        axes.vlines(session.licks[session.licks.bout_number == b].timestamps,bb,tt,alpha=1,linewidth=2,color = colors[np.mod(b,len(colors))])
    axes.plot(session.licks.groupby('bout_number').first().timestamps, tt*np.ones(np.shape(session.licks.groupby('bout_number').first().timestamps)), 'kv')
    axes.plot(session.licks.groupby('bout_number').last().timestamps, bb*np.ones(np.shape(session.licks.groupby('bout_number').first().timestamps)), 'k^')
    axes.vlines(session.licks[session.licks.bout_rewarded].timestamps,0.45,0.55,alpha=1,linewidth=2,color='r')
    axes.plot(session.rewards.timestamps,np.zeros(np.shape(session.rewards.timestamps.values))+0.5, 'ro', label='reward',markersize=10)
    axes.set_xlabel('time (s)',fontsize=16)
    axes.yaxis.set_tick_params(labelsize=16) 
    axes.xaxis.set_tick_params(labelsize=16)
    plt.tight_layout()
    
    def on_key_press(event):
        xStep = 20
        x = axes.get_xlim()
        xmin = x[0]
        xmax = x[1]
        if event.key=='<' or event.key==',' or event.key=='left': 
            xmin -= xStep
            xmax -= xStep
        elif event.key=='>' or event.key=='.' or event.key=='right':
            xmin += xStep
            xmax += xStep
        axes.set_xlim(xmin,xmax)
    kpid = fig.canvas.mpl_connect('key_press_event', on_key_press)
    return fig, axes


def get_bout_table(session):
    bout = session.licks.groupby('bout_number').apply(len).to_frame()
    bout = bout.rename(columns={0:"length"})
    bout['bout_rewarded'] = session.licks.groupby('bout_number').any('rewarded')['bout_rewarded']
    bout['bout_duration'] = session.licks.groupby('bout_number').last()['timestamps'] - session.licks.groupby('bout_number').first()['timestamps']
    bout['post_ili'] = session.licks.groupby('bout_number').last()['post_ili']
    bout['post_ili_from_start'] = session.licks.groupby('bout_number').last()['post_ili'] + bout['bout_duration']
    bout['pre_ili'] = session.licks.groupby('bout_number').first()['pre_ili']
    bout['pre_ili_from_start'] = session.licks.groupby('bout_number').first()['pre_ili'] + bout['bout_duration'].shift(1)
    return bout

def plot_bout_ili(bout,from_start=False):
    plt.figure()
    if from_start:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'], bins = 25,color='k',label='post-miss')
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'],bins=h[1], color='r',label='post-reward')
    else:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'], bins = 25,color='k',label='post-miss')
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'],bins=h[1], color='r',label='post-reward')
    plt.legend()
    plt.ylabel('count',fontsize=12)
    if from_start:
        plt.xlabel('InterLick (s), time from bout start', fontsize=12)
    else:
        plt.xlabel('InterLick (s), time from bout end', fontsize=12)
    ylims = plt.ylim()
    mean_all, mean_reward = get_bout_ili(bout, from_start=from_start)
    #if from_start:
    #    mean_all = np.nanmean(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'])
    #    mean_reward = np.nanmean(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'])
    #    #sem_all = bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'].sem()
    #    #sem_reward = bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'].sem()
    #    #n_all = len(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'])
    #    #n_reward = len(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'])
    #else:
    #    mean_all = np.nanmean(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'])
    #    mean_reward = np.nanmean(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'])
    #    #sem_all = bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'].std()
    #    #sem_reward = bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'].std()
    #    #n_all = len(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'])
    #    #n_reward = len(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'])
    plt.plot(mean_all, ylims[1], 'kv')    
    plt.plot(mean_reward, ylims[1], 'rv') 
    #plt.plot([mean_all-1.96*sem_all/n_all, mean_all+1.96*sem_all/n_all], [ylims[1], ylims[1]], 'k-')    
    #plt.plot([mean_reward-1.96*sem_reward/n_reward, mean_reward+1.96*sem_reward/n_reward], [ylims[1], ylims[1]], 'r-') 



def plot_bout_ili_current(bout,from_start=False):
    plt.figure()
    if from_start:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili_from_start'], bins = 25,color='b',label='post-miss miss',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili_from_start'],bins=h[1], color='m',label='post-reward miss',alpha=.25)
        plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili_from_start'],bins=h[1], color='k',label='post-miss hit',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili_from_start'],bins=h[1], color='r',label='post-reward hit',alpha=.25)
    else:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili'], bins = 25,color='b',label='post-miss miss',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili'],bins=h[1], color='m',label='post-reward miss',alpha=.25)
        plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili'],bins=h[1], color='k',label='post-miss hit',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili'],bins=h[1], color='r',label='post-reward hit',alpha=.25)
    plt.legend()
    plt.ylabel('count',fontsize=12)
    if from_start:
        plt.xlabel('InterLick (s), time from bout start', fontsize=12)
    else:
        plt.xlabel('InterLick (s), time from bout end', fontsize=12)
    ylims = plt.ylim()
    mean_miss, mean_reward = get_bout_ili_current(bout, from_start=from_start,current_hit=True)
    plt.plot(mean_miss, ylims[1], 'kv')    
    plt.plot(mean_reward, ylims[1], 'rv') 
    mean_miss, mean_reward = get_bout_ili_current(bout, from_start=from_start,current_hit=False)
    plt.plot(mean_miss, ylims[1], 'b^')    
    plt.plot(mean_reward, ylims[1], 'm^') 
 
def get_bout_ili(bout, from_start=False):
    if from_start:
        mean_miss = np.nanmean(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'])
        mean_reward = np.nanmean(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'])
    else:
        mean_miss = np.nanmean(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'])
        mean_reward = np.nanmean(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'])
    return mean_miss, mean_reward

def get_bout_ili_current(bout,from_start=False, current_hit=True):
    if from_start:
        start_str = 'post_ili_from_start'
    else:
        start_str = 'post_ili'
    if current_hit:
        mean_miss = np.nanmean( bout[(~bout['bout_rewarded'])&(bout['post_ili']<10)&(bout['bout_rewarded'].shift(-1))][start_str])
        mean_reward = np.nanmean( bout[(bout['bout_rewarded'])&(bout['post_ili']<10)&(bout['bout_rewarded'].shift(-1))][start_str])
    else:
        mean_miss = np.nanmean( bout[(~bout['bout_rewarded'])&(bout['post_ili']<10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))][start_str])
        mean_reward = np.nanmean( bout[(bout['bout_rewarded'])&(bout['post_ili']<10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))][start_str])  
    return mean_miss, mean_reward

def plot_bout_durations(bout):
    plt.figure()
    h = plt.hist(bout['length'],bins=np.max(bout['length']),color='k',label='All Bouts')
    plt.hist(bout[bout['bout_rewarded']]['length'],bins=h[1],color='r',label='Rewarded')
    plt.xlabel('# Licks in bout',fontsize=12)
    plt.ylabel('Count',fontsize=12)
    plt.legend()
    plt.gca().set_xticks(np.arange(0,np.max(bout['length']),5))

    plt.figure()
    h = plt.hist(bout['bout_duration'],bins=np.max(bout['length']),color='k',label='All Bouts')
    plt.hist(bout[bout['bout_rewarded']]['bout_duration'],bins=h[1],color='r',label='Rewarded')
    plt.xlabel('bout duration (s)',fontsize=12)
    plt.ylabel('Count',fontsize=12)
    plt.legend()
    plt.gca().set_xticks(np.arange(0,np.max(bout['bout_duration']),0.5))


def get_all_bout_table(IDS):
    session = ps.get_data(IDS[0])
    annotate_licks(session)
    all_bout = get_bout_table(session)
 
    for id in IDS[1:]:
        print(id)
        try:
            session = ps.get_data(id)
            annotate_licks(session)
            bout = get_bout_table(session)
            all_bout = pd.concat([all_bout, bout])    
        except:
            print(" crash")
    return all_bout    

def get_all_bout_statistics(IDS):
    durs = []
    for id in IDS:
        print(id)
        try:
            session = ps.get_data(id)
            annotate_licks(session)
            bout = get_bout_table(session) 
            my_durs = np.concatenate([get_bout_ili(bout, from_start=False), get_bout_ili(bout,from_start=True), get_bout_ili_current(bout, from_start=False,current_hit=False),get_bout_ili_current(bout, from_start=True,current_hit=False),get_bout_ili_current(bout, from_start=False,current_hit=True),get_bout_ili_current(bout, from_start=True,current_hit=True)])
            durs.append(my_durs)
        except:
            print(" crash")
    return durs    

def plot_all_bout_statistics(durs,all_bout=None):
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

def plot_all_bout_statistics_current(durs,all_bout=None):
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
    

