import psy_tools as ps
from alex_utils import *
import numpy as np
import matplotlib.pyplot as plt


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
    d = session.licks['pre-ili'][session.licks.rewarded]
    hits = len(d[(d>.7)& (d<10)].values)
    d = session.licks['pre-ili']
    total = len(d[(d>.7)& (d<10)].values)   
    return total, hits

def annotate_licks(session):
    # ili
    # rewarded
    # consumption
    licks = session.licks
    licks['pre-ili'] = np.concatenate([[np.nan],np.diff(licks.timestamps.values)])
    licks['post-ili'] = np.concatenate([np.diff(licks.timestamps.values),[np.nan]])
    licks['rewarded'] = False
    for index, row in session.rewards.iterrows():
        mylick = np.where(licks.timestamps <= row.timestamps)[0][-1]
        licks.at[mylick,'rewarded'] = True

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
    d = session.licks['pre-ili'][session.licks.rewarded]
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
    d = session.licks['pre-ili'][session.licks.rewarded]
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
 
def get_mean_lick_distribution(session):
    licks = session.licks.timestamps.values
    diffs = np.diff(licks)
    good_diffs = diffs[(diffs<10) & (diffs > 0.75)]
    return np.mean(good_diffs)


