import psy_tools as ps
import psy_general_tools as pgt
import psy_timing_tools as pt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.ion()


################# Licking wrt flash cycle
def plot_sessions(ids,directory=None):
    a,c = get_sessions(ids)
    plot_licks_by_flash(a,c,filename=directory)

def plot_session(id,directory=None):
    all_bout_start_times, change_bout_start_times = get_session_licks(id)
    plot_licks_by_flash(all_bout_start_times,change_bout_start_times,title_str=str(id),filename=directory)

def annotate_licks_by_flash(session):
    session.stimulus_presentations['first_lick'] = [x[0] if (len(x) > 0) else np.nan for x in session.stimulus_presentations.licks]
    session.stimulus_presentations['last_lick'] = [x[-1] if (len(x) > 0) else np.nan for x in session.stimulus_presentations.licks]
    session.stimulus_presentations['last_lick_back_1'] = session.stimulus_presentations['last_lick'].shift(1)
    session.stimulus_presentations['first_lick_ili'] = session.stimulus_presentations['first_lick'] - session.stimulus_presentations['last_lick_back_1']
    session.stimulus_presentations['bout_start'] = (~np.isnan(session.stimulus_presentations['first_lick']) & (np.isnan(session.stimulus_presentations['first_lick_ili']) | (session.stimulus_presentations['first_lick_ili'] > 0.7)))
    session.stimulus_presentations['lick_time'] = session.stimulus_presentations['first_lick'] - session.stimulus_presentations['start_time']

def plot_licks_by_flash(all_bout_start_times, change_bout_start_times,title_str="",filename=None):
    plt.figure()
    plt.hist(all_bout_start_times,45,color='gray',label='All Flashes')
    plt.hist(change_bout_start_times,45,color='black',label='Change Flashes')
    plt.xlim(0,.75)
    plt.xlabel('Time in Flash Cycle')
    plt.ylabel('count')
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    if type(filename) is not type(None):
        plt.savefig(filename+"_flash_cycle.svg")

def get_session_licks(id,return_session=False):
    session = pgt.get_data(id)
    annotate_licks_by_flash(session)
    all_bout_start_times = session.stimulus_presentations.lick_time[~np.isnan(session.stimulus_presentations.lick_time)&(session.stimulus_presentations['bout_start'])]
    change_bout_start_times = session.stimulus_presentations.lick_time[(~np.isnan(session.stimulus_presentations.lick_time))&(session.stimulus_presentations['change'])&(session.stimulus_presentations['bout_start'])]
    if return_session:
        return all_bout_start_times.values, change_bout_start_times.values,session
    else:
        return all_bout_start_times.values, change_bout_start_times.values

def get_sessions(ids):
    all_times = []
    change_times =[]
    for id in ids:
        print(id)
        a,c = get_session_licks(id)
        all_times.append(a)
        change_times.append(c)
    return np.concatenate(all_times), np.concatenate(change_times)

def build_session_table(ids,fit_directory=None):
    df = pd.DataFrame(data={'peakiness':[],'hit_percentage':[],'hit_count':[],'licks':[],'task_index':[]})
    for id in ids:
        print(id)
        a,c,session = get_session_licks(id,return_session=True)       
        counts,edges = np.histogram(a,bins=45)
        var = np.var(counts)
        hit_count = np.sum(session.trials.hit)
        licks = np.sum(session.stimulus_presentations.bout_start)
        hit_percentage = hit_count/licks
        d = {'peakiness':var/np.mean(counts),'hit_percentage':hit_percentage,'hit_count':hit_count,'licks':licks,'task_index':ps.get_timing_index(id,fit_directory)}
        df = df.append(d,ignore_index=True)
    return df
