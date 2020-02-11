import psy_tools as ps
import psy_general_tools as pgt
import psy_timing_tools as pt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm

################# Licking wrt flash cycle
def plot_all_sessions(ids,directory):
    for id in tqdm(ids):
        plot_session(id,directory=directory+str(id))
        plt.close('all')

def plot_all_mouse_sessions(mice_ids, directory):
    for mouse in tqdm(mice_ids):
        try:
            mice_ids = pgt.get_mice_sessions(mouse)
            plot_sessions(mice_ids, directory=directory+"Mouse_"+str(mouse),return_counts=True)
        except:
            print(f"crash {mouse}")
        plt.close('all')

def plot_sessions(ids,directory=None,return_counts=False):
    if return_counts:
        a,c,na,nc = get_sessions(ids,return_counts=True)
    else:
        a,c = get_sessions(ids)
    plot_licks_by_flash(a,c,filename=directory)
    if return_counts:
        plot_lick_fraction_by_flash(a,c,na,nc,filename=directory)
        plot_lick_fraction_normalized_by_flash(a,c,na,nc,filename=directory)
        return a,c,na,nc
    else:
        return a,c

def plot_session(id,directory=None):
    all_bout_start_times, change_bout_start_times,na,nc = get_session_licks(id,return_counts=True)
    plot_licks_by_flash(all_bout_start_times,change_bout_start_times,title_str=str(id),filename=directory)
    plot_lick_fraction_by_flash(all_bout_start_times,change_bout_start_times,na,nc,title_str=str(id),filename=directory)
    plot_lick_fraction_normalized_by_flash(all_bout_start_times,change_bout_start_times,na,nc,title_str=str(id),filename=directory)

def annotate_licks_by_flash(session):
    session.stimulus_presentations['first_lick'] = [x[0] if (len(x) > 0) else np.nan for x in session.stimulus_presentations.licks]
    session.stimulus_presentations['last_lick'] = [x[-1] if (len(x) > 0) else np.nan for x in session.stimulus_presentations.licks]
    session.stimulus_presentations['last_lick_back_1'] = session.stimulus_presentations['last_lick'].shift(1)
    session.stimulus_presentations['first_lick_ili'] = session.stimulus_presentations['first_lick'] - session.stimulus_presentations['last_lick_back_1']
    session.stimulus_presentations['bout_start'] = (~np.isnan(session.stimulus_presentations['first_lick']) & (np.isnan(session.stimulus_presentations['first_lick_ili']) | (session.stimulus_presentations['first_lick_ili'] > 0.7)))
    session.stimulus_presentations['lick_time'] = session.stimulus_presentations['first_lick'] - session.stimulus_presentations['start_time']

def plot_licks_by_flash(all_bout_start_times, change_bout_start_times,title_str="",filename=None):
    plt.figure()
    plt.hist(all_bout_start_times,45,color='gray',label='Non-Change Flashes',alpha=0.7)
    plt.hist(change_bout_start_times,45,color='black',label='Change Flashes',alpha=1)
    plt.xlim(0,.75)
    plt.xlabel('Time in Flash Cycle',fontsize=12)
    plt.ylabel('count',fontsize=12)
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    if type(filename) is not type(None):
        plt.savefig(filename+"_flash_cycle.svg")

def plot_lick_fraction_by_flash(all_bout_start_times, change_bout_start_times,num_all,num_change, title_str="",filename=None,fs1=12,fs2=12):
    plt.figure()
    all_counts,all_edges = np.histogram(all_bout_start_times,45)
    cweights = np.ones_like(change_bout_start_times)/float(num_change)
    aweights = np.ones_like(all_bout_start_times)/float(num_all)
    plt.hist(change_bout_start_times, bins=all_edges, color='black',alpha=1,label='Change',  weights=cweights)
    plt.hist(all_bout_start_times, bins=all_edges, color='gray',alpha=0.7,label='Non-Change',weights=aweights)
    plt.xlabel('Time in Flash Cycle',fontsize=fs1)
    plt.ylabel('Lick Fraction',fontsize=fs1)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.xlim(0,.75)
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    if type(filename) is not type(None):
        plt.savefig(filename+"_lick_fraction_flash_cycle.svg")

def plot_lick_fraction_normalized_by_flash(all_bout_start_times, change_bout_start_times,num_all,num_change, title_str="",filename=None):
    plt.figure()
    all_counts,all_edges = np.histogram(all_bout_start_times,45)
    cweights = np.ones_like(change_bout_start_times)/float(len(change_bout_start_times))
    aweights = np.ones_like(all_bout_start_times)/float(len(all_bout_start_times))
    plt.hist(change_bout_start_times, bins=all_edges, color='black',alpha=1,label='Change',  weights=cweights)
    plt.hist(all_bout_start_times, bins=all_edges, color='gray',alpha=0.7,label='Non-Change',weights=aweights)
    plt.xlabel('Time in Flash Cycle',fontsize=12)
    plt.ylabel('Lick Fraction',fontsize=12)
    plt.xlim(0,.75)
    plt.title(title_str)
    plt.legend()
    plt.tight_layout()
    if type(filename) is not type(None):
        plt.savefig(filename+"_lick_fraction_normalized_flash_cycle.svg")

def get_session_licks(id,return_session=False,return_counts=False):
    session = pgt.get_data(id)
    annotate_licks_by_flash(session)
    all_bout_start_times = session.stimulus_presentations.lick_time[~np.isnan(session.stimulus_presentations.lick_time)&(~session.stimulus_presentations['change'])&(session.stimulus_presentations['bout_start'])]
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
    for id in tqdm(ids):
        try:
            if return_counts:
                a,c,na,nc = get_session_licks(id,return_counts=True)
            else:
                a,c = get_session_licks(id)
        except:
            print(f"crash {id}")
        else:
            if return_counts:
                numall += na
                numchange += nc
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
