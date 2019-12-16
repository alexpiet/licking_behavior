import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
from importlib import reload
plt.ion()
import numpy as np
import pandas as pd
import psy_glm_tools as pg
import hierarchical_boot as hb
from tqdm import tqdm
import seaborn as sns



def block_to_mean_dff(df):
    return df['mean_response'][0:8].values

def get_cell_psth(cell,session):
    fr = session.flash_response_df.query('pref_stim')
    cell_fr = fr.query('cell_specimen_id == @cell').groupby('block_index').apply(block_to_mean_dff)
    cell_fr = [x for x in list(cell_fr) if len(x) == 8]  
    if len(cell_fr) > 0:
        cell_fr =  np.mean(np.vstack(cell_fr),0)
    else:
        cell_fr = []
    trial_fr = session.trial_response_df.query('pref_stim & cell_specimen_id == @cell')['dff_trace'].mean() 
    trial_fr_timestamps = session.trial_response_df.iloc[0].dff_trace_timestamps - session.trial_response_df.iloc[0].change_time
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],'mean_response_trace':[],'dff_trace':[],'dff_trace_timestamps':[]})
    d = {
        'ophys_experiment_id':session.metadata['ophys_experiment_id'],
        'stage':session.metadata['stage'],
        'cell':cell,
        'imaging_depth':session.metadata['imaging_depth'],
        'mean_response_trace':cell_fr,
        'dff_trace':trial_fr,
        'dff_trace_timestamps':trial_fr_timestamps}
    df = df.append(d,ignore_index=True)
    return df

def get_session_psth(session):
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],'mean_response_trace':[],'dff_trace':[],'dff_trace_timestamps':[]})
    cellids = session.flash_response_df['cell_specimen_id'].unique()
    for index, cell in enumerate(cellids):
        cell_df = get_cell_psth(cell,session)
        df = df.append(cell_df,ignore_index=True)
    return df

def get_average_psth(session_ids):
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],'mean_response_trace':[],'dff_trace':[],'dff_trace_timestamps':[]})
    for index, session_id in tqdm(enumerate(session_ids)):
        session = ps.get_data(session_id)
        session_df = get_session_psth(session)
        df = df.append(session_df,ignore_index=True)
    df = pg.annotate_stage(df)
    return df 

def get_all_df(path='/home/alex.piet/codebase/allen/all_slc_exp_df.csv',force_recompute=False):
    try:
        all_df =pd.read_csv(filepath_or_buffer = path)
    except:
        if force_recompute:
            all_df = get_average_psth(ps.get_slc_session_ids())
            all_df = pg.annotate_stage(all_df)
            all_df.to_csv(path_or_buf=path)
        else:
            raise Exception('file not found: ' + path)
    return all_df

def compare_groups(all_df,queries, labels):
    dfs=[]
    for q in queries:
        dfs.append(all_df.query(q))
    plot_mean_trace(dfs,labels)

def plot_mean_trace(dfs, labels):    
    plt.figure()
    colors = sns.color_palette(n_colors=2)
    for index, df in enumerate(dfs):
        plt.plot(df['dff_trace'].mean()-np.min(df['dff_trace'].mean()), color=colors[index], alpha=0.5,label=labels[index])
    plt.ylim(0,.2)
    plt.ylabel('Average PSTH (df/f)')
    plt.xlabel('# Repetition in Block')
    plt.legend()





