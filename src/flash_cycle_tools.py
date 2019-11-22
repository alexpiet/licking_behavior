import psy_tools as ps
import psy_general_tools as pgt
import psy_timing_tools as pt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
plt.ion()


################# Licking wrt flash cycle
def plot_sessions(ids,directory=None,return_counts=False):
    if return_counts:
        a,c,na,nc = get_sessions(ids,return_counts=True)
    else:
        a,c = get_sessions(ids)
    plot_licks_by_flash(a,c,filename=directory)
    if return_counts:
        plot_lick_fraction_by_flash(a,c,na,nc,filename=directory)

def plot_session(id,directory=None):
    all_bout_start_times, change_bout_start_times,na,nc = get_session_licks(id,return_counts=True)
    plot_licks_by_flash(all_bout_start_times,change_bout_start_times,title_str=str(id),filename=directory)
    plot_lick_fraction_by_flash(all_bout_start_times,change_bout_start_times,na,nc,title_str=str(id),filename=directory)

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

def plot_lick_fraction_by_flash(all_bout_start_times, change_bout_start_times,num_all,num_change, title_str="",filename=None):
    plt.figure()
    all_counts,all_edges = np.histogram(all_bout_start_times,45)
    change_counts,change_edges = np.histogram(change_bout_start_times,45)
    change_centers = change_edges[0:-1] + np.diff(change_edges)/2
    all_centers = all_edges[0:-1] + np.diff(all_edges)/2
    plt.bar(range(0,45), change_counts/num_change,width=1,color='black',alpha=1, label='Change')
    plt.bar(range(0,45), all_counts/num_all,width=1,color='gray',label='All',alpha=0.7)
    #plt.xticks(range(0,45), np.round(all_centers,2).astype(str))
    plt.xlabel('Time in Flash Cycle')
    plt.ylabel('Lick Fraction')
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    if type(filename) is not type(None):
        plt.savefig(filename+"_lick_fraction_flash_cycle.svg")

def get_session_licks(id,return_session=False,return_counts=False):
    session = pgt.get_data(id)
    annotate_licks_by_flash(session)
    all_bout_start_times = session.stimulus_presentations.lick_time[~np.isnan(session.stimulus_presentations.lick_time)&(session.stimulus_presentations['bout_start'])]
    change_bout_start_times = session.stimulus_presentations.lick_time[(~np.isnan(session.stimulus_presentations.lick_time))&(session.stimulus_presentations['change'])&(session.stimulus_presentations['bout_start'])]
    if return_session:
        return all_bout_start_times.values, change_bout_start_times.values,session
    elif return_counts:
        return all_bout_start_times.values, change_bout_start_times.values, len(session.stimulus_presentations), np.sum(session.stimulus_presentations.change)
    else:
        return all_bout_start_times.values, change_bout_start_times.values

def get_sessions(ids,return_counts=False):
    all_times = []
    change_times =[]
    numall = 0
    numchange = 0
    for id in ids:
        print(id)
        if return_counts:
            a,c,na,nc = get_session_licks(id,return_counts=True)
            numall += na
            numchange += nc
        else:
            a,c = get_session_licks(id)
        all_times.append(a)
        change_times.append(c)
    if return_counts:
        return np.concatenate(all_times), np.concatenate(change_times), numall, numchange
    else:
        return np.concatenate(all_times), np.concatenate(change_times)

def build_session_table(ids,fit_directory=None):
    df = pd.DataFrame(data={'peakiness':[],'hit_percentage':[],'hit_count':[],'licks':[],'task_index':[],'mean_dprime':[]})
    for id in ids:
        print(id)
        a,c,session = get_session_licks(id,return_session=True)       
        counts,edges = np.histogram(a,bins=45)
        var = np.var(counts)
        hit_count = np.sum(session.trials.hit)
        licks = np.sum(session.stimulus_presentations.bout_start)
        hit_percentage = hit_count/licks
        dprime = session.get_performance_metrics()['mean_dprime']
        d = {'peakiness':var/np.mean(counts),'hit_percentage':hit_percentage,'hit_count':hit_count,'licks':licks,'task_index':ps.get_timing_index(id,fit_directory),'mean_dprime':dprime}
        df = df.append(d,ignore_index=True)
    return df
