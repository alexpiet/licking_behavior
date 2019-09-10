import psy_tools as ps
from alex_utils import *
import numpy as np
import matplotlib.pyplot as plt



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

def plot_all_mouse_durations(all_durs):
    plt.figure()
    for dur in all_durs:
        if len(dur) == 4:
            plt.plot(dur)

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

# Make Figure of distribution of licks
def plot_lick_distribution(session):
    annotate_licks(session)
    licks = session.licks.timestamps.values
    diffs = np.diff(licks)
    plt.figure()
    plt.hist(diffs[diffs<10],50,label='All')
    ax = plt.gca()
    ax.axvline(0.75,linestyle='--',color='k')
    plt.ylabel('count')
    plt.xlabel('InterLick (s)')
    plt.ylim([0,100])
    plt.title(str(session.metadata['mouse_id'])+" "+session.metadata['stage'])
    m = get_mean_lick_distribution(session)
    ax.axvline(m,linestyle='--',color='r')
    d = session.licks['pre-ili'][session.licks.rewarded]
    plt.hist(d[(d>.7)&(d<10)],30,label='Hits')
    plt.legend()

def get_mean_lick_distribution(session):
    licks = session.licks.timestamps.values
    diffs = np.diff(licks)
    good_diffs = diffs[(diffs<10) & (diffs > 0.75)]
    return np.mean(good_diffs)


