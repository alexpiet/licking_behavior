import psy_tools as ps
from alex_utils import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn


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
    annotate_licks(session)
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

def get_chronometric(session,nbins=50, directory=None):
    annotate_licks(session)
    d = session.licks['pre_ili']
    dr = session.licks['pre_ili'][session.licks.rewarded]
    h= np.histogram(d[(d>.7)&(d<10)],nbins)
    h1= np.histogram(dr[(dr>.7)&(dr<10)],bins=h[1])
    centers = np.diff(h[1]) + h[1][0:-1]
    fig, ax = plt.subplots(2,1)
    chrono = np.array(h1[0])/h[0]
    #ax.plot(centers[centers > .7], (h1[0]-h2[0])[centers > .7]  ,'k-')
    ax[0].plot(centers[centers > .7], h[0][centers > .7], 'k')
    ax[0].plot(centers[centers > .7], h1[0][centers > .7], 'r')
    ax[0].set_ylabel('% Hit')
    ax[0].set_xlabel('InterLick (s)')
    ax[1].plot(centers[centers > .7], chrono[centers > .7], 'k')
    ax[1].set_ylabel('% Hit')
    ax[1].set_xlabel('InterLick (s)')
    plt.tight_layout()
    if type(directory) is not type(None):
        id = session.metadata['ophys_experiment_id']
        plt.savefig(directory+str(id)+"_chronometric.png")
 
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
    for index, row in session.stimulus_presentations.iterrows():
        if not row.omitted:
            axes.axvspan(row.start_time,row.stop_time, alpha=0.2,color='k', label='flash')
        if row.change:
            axes.axvspan(row.start_time,row.stop_time, alpha=0.6,color='c', label='change flash')
    bouts = session.licks.bout_number.unique()
    for b in bouts:
        axes.vlines(session.licks[session.licks.bout_number == b].timestamps,0.8,0.9,alpha=1,linewidth=2,color = colors[np.mod(b,len(colors))])
    #axes.vlines(session.licks.timestamps,0.9,.95,alpha=1,linewidth=2)
    axes.vlines(session.licks[session.licks.bout_rewarded].timestamps,0.75,.8,alpha=1,linewidth=2,color='r')
    axes.plot(session.rewards.timestamps,np.zeros(np.shape(session.rewards.timestamps.values))+0.8, 'ro', label='reward',markersize=10)
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
    return bout

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








