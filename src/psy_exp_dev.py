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

def get_all_df(path='/home/alex.piet/Desktop/all_slc_exp_df.csv',force_recompute=False):
    try:
        all_df =pd.read_csv(filepath_or_buffer = path)
    except:
        if force_recompute:
            all_df = get_average_psth(ps.get_slc_session_ids())
            all_df = pg.annotate_stage(all_df)
            all_df.to_csv(path_or_buf=path)
        else:
            error('file not found: ' + path)
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




all_df = get_all_df()
compare_groups(all_df,['active','not active'],['Active','Passive'])
compare_groups(all_df,['active & imaging_depth == 175','not active & imaging_depth == 175'],['Active 175','Passive 175'])
compare_groups(all_df,['active & imaging_depth == 375','not active & imaging_depth == 375'],['Active 375','Passive 375'])

compare_groups(all_df,['image_set == "A"','image_set == "B"'],['A','B'])

compare_groups(all_df,['stage_num == "1"','stage_num == "4"'],['1','4'])
compare_groups(all_df,['stage_num == "4"','stage_num == "6"'],['4','6'])

plt.figure()
cm_4_175 = (np.mean(cell_psths_4_175,0)-np.mean(cell_psths_6_175,0))/(np.mean(cell_psths_4_175,0)+np.mean(cell_psths_6_175,0))
cm_4_375 = (np.mean(cell_psths_4_375,0)-np.mean(cell_psths_6_375,0))/(np.mean(cell_psths_4_375,0)+np.mean(cell_psths_6_375,0))
plt.plot(cm_4_175,color=colors[0],label='CM 175')
plt.plot(cm_4_375,color=colors[1],label='CM 375')






