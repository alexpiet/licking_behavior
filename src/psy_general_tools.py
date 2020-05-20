import os
import numpy as np
import pandas as pd
import psy_tools as ps
import matplotlib.pyplot as plt
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc
from visual_behavior.translator.allensdk_sessions import sdk_utils
from visual_behavior.ophys.response_analysis import response_processing as rp
from visual_behavior.ophys.response_analysis import utilities as ru
'''
This is a set of general purpose functions for interacting with the SDK
Alex Piet, alexpiet@gmail.com
11/5/2019
updated 01/22/2020
updated 04/07/2020

'''

MANIFEST_PATH = os.path.join("/home/alex.piet/codebase/behavior/manifest/", "manifest.json")

def add_block_index_to_stimulus_response_df(session):
    # Both addsin place
    session.stimulus_presentations['block_index'] = session.stimulus_presentations.change.cumsum() 
    # Have to merge into flash_response_df
    session.flash_response_df = session.flash_response_df.merge(session.stimulus_presentations.reset_index()[['stimulus_presentations_id','block_index','start_time','image_name']],on='stimulus_presentations_id')

def get_stimulus_response_df(session):
    params = {
        "window_around_timepoint_seconds": [-0.5, 0.75],
        "response_window_duration_seconds": 0.75,
        "baseline_window_duration_seconds": 0.25,
        "ophys_frame_rate": 31,
    }
    session.flash_response_df = rp.stimulus_response_df(rp.stimulus_response_xr(session,response_analysis_params=params))
    add_block_index_to_stimulus_response_df(session)

def get_trial_response_df(session):
    session.trial_response_df = rp.trial_response_df(rp.trial_response_xr(session))

def get_data(bsid,OPHYS=True):
    '''
        Loads data from SDK interface
        ARGS: bsid to load
        if OPHYS is true, loads data from the OPHYS api instead
    '''

    if OPHYS:
        session = get_data_from_oeid(sdk_utils.get_ophys_experiment_id_from_behavior_session_id(bsid,get_cache()))
        clean_session(session)
    else:
        session = get_training_data(bsid)
    return session

def clean_session(session):
    '''
        SDK PATCH
    '''
    sdk_utils.add_stimulus_presentations_analysis(session)

def clean_training_session(session):
    sdk_utils.add_stimulus_presentations_analysis(session,add_running_speed=False)

def get_data_from_bsid(bsid):
    '''
        Loads data from SDK interface
        ARGS: behavior_session_id to load
    '''
   
    cache = get_cache()
    return cache.get_behavior_session_data(bsid)

def get_data_from_oeid(oeid):
    '''
        Loads data from SDK interface
        ARGS: ophys_experiment_id to load
    '''
    cache = get_cache()
    return cache.get_session_data(oeid)

def get_training_data(bsid):
    session = get_data_from_bsid(bsid)

    print('WARNING SUPER SDK BUG HACK')
    early_training = (session.metadata['session_type'][0:8] == 'TRAINING') and (int(session.metadata['session_type'][9])<5)
    if early_training:
        print('    Using first stimulus for hack')
        first_stim = session.stimulus_presentations.iloc[0].start_time
    else:
        print('    Using first 300 stimulus for hack')
        first_stim = session.stimulus_presentations[session.stimulus_presentations.start_time > 300].iloc[0].start_time

    offset = session.trials.iloc[0]['start_time'] - first_stim
    session.trials['change_time'] = session.trials['change_time'] + offset
    session.stimulus_presentations['start_time'] = session.stimulus_presentations['start_time'] + offset
    session.stimulus_presentations['stop_time'] = session.stimulus_presentations['stop_time'] + offset

    clean_training_session(session)
    return session

def test_get_training_data(bsid):
    session = get_data_from_bsid(bsid)
    count = 0
    try:
        stim = session.stimulus_presentations
        trials = session.trials
        rewards = session.rewards
        licks = session.licks
        meta = session.metadata
        count = 1
        sdk_utils.add_stimulus_presentations_analysis(session,add_running_speed=False)
        count = 2
        running_speed = session.running_speed
    except:
        return count
    else:
        return True

def moving_mean(values, window):
    '''
        Computes the moving mean of the series in values, with a square window of width window
    '''
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm
 
def get_stage(oeid):
    '''
        Returns the stage name as a string 
        ARGS: ophys_experiment_id
    '''
    ophys_experiments = cache.get_experiment_table()
    return ophys_experiments.loc[oeid]['session_type']

def get_intersection(list_of_ids):
    '''
        Returns the intersection of values in the list
    '''
    return reduce(np.intersect1d,tuple(list_of_ids))

def get_slc_session_ids():
    '''
        Returns an array of the behavior_session_ids from SLC mice 
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('cre_line == "Slc17a7-IRES2-Cre"').index)
    return session_ids

def get_vip_session_ids():
    '''
        Returns an array of the behavior_session_ids from VIP mice 
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('cre_line == "Vip-IRES-Cre"').index)
    return session_ids
   
def get_session_ids():
    '''
        Returns an array of the behavior_session_ids
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.index)
    return session_ids

def get_active_ids():
    '''
        Returns an array of the behavior_session_ids from active sessions
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('active').index)
    return session_ids

def get_passive_ids():
    '''
        Returns an array of the behavior_session_ids from passive sessions
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('not active').index)
    return session_ids

def get_A_ids():
    '''
        Returns an array of the behavior_session_ids from sessions using image set A
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('image_set == "A"').index)
    return session_ids

def get_B_ids():
    '''
        Returns an array of the behavior_session_ids from sessions using image set B
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('image_set == "B"').index)
    return session_ids

def get_active_A_ids():
    '''
        Returns an array of the behavior_session_ids from active sessions using image set A
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('active & image_set == "A"').index)
    return session_ids

def get_active_B_ids():
    '''
        Returns an array of the behavior_session_ids from active sessions using image set B
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('active & image_set == "B"').index)
    return session_ids

def get_stage_ids(stage):
    '''
        Returns an array of the behavior_session_ids in stage 
    '''
    manifest = get_manifest()
    stage = str(stage)
    session_ids = np.unique(manifest.query('session_type.str[6] == @stage').index)
    return session_ids

def get_layer_ids(depth):
    '''
        Returns an array of the behavior_session_ids imaged at depth
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('imaging_depth == @depth').index)
    return session_ids

def get_mice_ids():
    '''
        Returns an array of the donor_ids
    '''
    return get_donor_ids()

def get_donor_ids():
    '''
        Returns an array of the donor_ids
    '''
    manifest = get_manifest()
    mice_ids = np.unique(manifest.donor_id.values)
    return mice_ids

def get_mice_sessions(donor_id):
    '''
        Returns an array of the behavior_session_ids by mouse donor_id
    '''
    mouse_manifest = get_mouse_manifest(donor_id)
    return np.array(mouse_manifest.index)

#####################################################################################

def get_training_manifest(force_recompute=False):
    '''
        Returns a dataframe of all behavior sessions that satisfy the optimal arguments
    '''
    if force_recompute:
        return compute_training_manifest()
    elif 'training_manifest' in globals():
        return training_manifest
    else:
        return compute_training_manifest()

def compute_training_manifest():
    '''
        Computes and returns dataframe of all behavior sessions
    '''

    # Get list of mice from ophys_manifest
    manifest = get_manifest()
    mice_ids = manifest.donor_id.unique()

    # Get full list of sessions from LIMS
    cache = get_cache()
    behavior_sessions = cache.get_behavior_session_table()

    # For each mouse, filter behavior_sessions, append to manifest?
    t_manifest = behavior_sessions[behavior_sessions.donor_id.isin(mice_ids)].copy()
    t_manifest.sort_index(inplace=True)
    t_manifest.drop(columns=['foraging_id','sex','full_genotype','reporter_line'],inplace=True)
    t_manifest =t_manifest[~t_manifest.session_type.isnull()]

    # Make nice cre-line
    t_manifest['cre_line'] = [x[-1] for x in t_manifest.driver_line]
 
    # Mark training sessions
    t_manifest['ophys'] = [x[0:5] =='OPHYS' for x in t_manifest.session_type]

    # Build list of stage
    t_manifest['stage'] = [x[1][6] if x[0] else x[1][9] for x in zip(t_manifest.ophys, t_manifest.session_type)]  

    # Build list of active sessions
    t_manifest['active'] = [(not x[0]) or (x[1] in ['0','1','3','4','6']) for x in zip(t_manifest.ophys, t_manifest.stage)]
   
    # Filter out bad OPHYS sessions 
    t_manifest['good'] = [True if not x[0] else True if x[1] == '0' else x[2] for x in zip(t_manifest.ophys,t_manifest.stage,t_manifest.index.isin(manifest.index))]
    t_manifest = t_manifest.query('good').copy().drop(columns=['good'])
    
    # Add absolute training numbers
    t_manifest['imaging'] = t_manifest.ophys & (t_manifest.stage >= "1")
    t_manifest['session_number'] = t_manifest.groupby('donor_id').cumcount()
    
    t_manifest['tmp'] = t_manifest.groupby(['donor_id','imaging']).cumcount()
    t_manifest['pre_ophys_number'] = t_manifest.groupby(['donor_id','imaging']).cumcount(ascending=False)
    t_manifest['pre_ophys_number'] = t_manifest['pre_ophys_number']+1
    t_manifest.loc[t_manifest['imaging'],'pre_ophys_number'] = -t_manifest[t_manifest['imaging']]['tmp']
    t_manifest= t_manifest.drop(columns=['tmp'])

    t_manifest = t_manifest.query('(ophys) or (not ophys and stage > "2")')

    # Cache manifest as global manifest
    global training_manifest
    training_manifest = t_manifest

    return t_manifest


#####################################################################################   
def get_manifest(require_cell_matching=False,require_full_container=False,require_exp_pass=True,force_recompute=False,include_mesoscope=False):
    '''
        Returns a dataframe of all the ophys_sessions that satisfy the optional arguments 
    '''
    if include_mesoscope:
        raise Exception('Not Implemented')

    if force_recompute:
        return compute_manifest(require_cell_matching=require_cell_matching, require_full_container=require_full_container,require_exp_pass=require_exp_pass,include_mesoscope=include_mesoscope)
    elif 'behavior_manifest' in globals():
        return behavior_manifest
    else:
        return compute_manifest(require_cell_matching=require_cell_matching, require_full_container=require_full_container,require_exp_pass=require_exp_pass,include_mesoscope=include_mesoscope)

def compute_manifest(require_cell_matching=False,require_full_container=False,require_exp_pass=True,include_mesoscope=False):
    '''
        Returns a dataframe which is the list of all sessions in the current cache
    '''
    if include_mesoscope:
       raise Exception('Not Implemented')

    cache = get_cache()
    ophys_session_filters = sdk_utils.get_filtered_sessions_table(cache, require_cell_matching=require_cell_matching,require_full_container=require_full_container,require_exp_pass=require_exp_pass)
    manifest = ophys_session_filters.reset_index().set_index('behavior_session_id')

    # make nice cre_line
    drivers = manifest.driver_line
    cre = [x[-1] for x in drivers]
    manifest['cre_line'] = cre
    
    # convert specimen ids to donor_ids
    manifest['donor_id'] = [sdk_utils.get_donor_id_from_specimen_id(x,cache) for x in manifest['specimen_id'].values]
    
    # Build list of active sessions
    manifest['active'] =  manifest['session_type'].isin(['OPHYS_1_images_A', 'OPHYS_3_images_A', 'OPHYS_4_images_A',
        'OPHYS_6_images_A',  'OPHYS_1_images_B', 'OPHYS_3_images_B', 'OPHYS_4_images_B', 'OPHYS_6_images_B'])

    # get container ID, and imaging_depth   
    ophys_experiments = cache.get_experiment_table()
    ophys_experiments = ophys_experiments.reset_index().set_index('behavior_session_id')
    manifest['imaging_depth'] = [ophys_experiments.loc[x]['imaging_depth'] for x in manifest.index]
    manifest['container_id'] = [ophys_experiments.loc[x]['container_id'] for x in manifest.index] 

    # get image set
    manifest['image_set'] = [manifest.loc[x]['session_type'][15] for x in manifest.index]

    # Clean up columns    
    manifest = manifest.drop(columns=['in_experiment_table','in_bsession_table','good_project_code','good_session','good_exp_workflow','good_container_workflow','session_name'])

    # Annotate what the training image set, and the numerical stage is for ease of use later
    manifest['trained_A'] = manifest.session_type.isin(['OPHYS_1_images_A','OPHYS_3_images_A','OPHYS_4_images_B','OPHYS_6_images_B'])
    manifest['trained_B'] = manifest.session_type.isin(['OPHYS_1_images_B','OPHYS_3_images_B','OPHYS_4_images_A','OPHYS_6_images_A'])
    manifest['stage'] = manifest.session_type.str[6]

    # Cache manifest as global manifest
    global behavior_manifest
    behavior_manifest = manifest

    return manifest

#####################################################################################   

def get_mouse_training_manifest(donor_id):
    '''
        Returns a dataframe containing all behavior_sessions for this donor_id
    '''
    t_manifest = get_training_manifest()
    mouse_t_manifest = t_manifest.query('donor_id == @donor_id').copy()
    return mouse_t_manifest
    
def get_mouse_manifest(donor_id):
    '''
        Returns a dataframe containing all ophys_sessions for this donor_id
    '''
    manifest = get_manifest()
    mouse_manifest =  manifest.query('donor_id ==@donor_id').copy()
    mouse_manifest = mouse_manifest.sort_values(by='date_of_acquisition')
    return mouse_manifest
    
def get_cache():
    '''
        Returns the SDK cache
    '''
    return bpc.from_lims(manifest=MANIFEST_PATH)

def get_experiment_table():
    cache = get_cache()
    return cache.get_experiment_table()

def get_ophys_sessions():
    cache = get_cache()
    return cache.get_session_table()

def get_behavior_sessions():
    cache = get_cache()
    return cache.get_behavior_session_table()

#####################################################################################   

def load_mouse(mouse):
    '''
        Takes a mouse donor_id, returns a list of all sessions objects, their IDS, and whether it was active or not. 
        no matter what, always returns the behavior_session_id for each session.    
    '''

    # Get mouse_manifest
    mouse_manifest = get_mouse_manifest(mouse)

    # Load the sessions 
    sessions = []
    IDS = []
    active =[]
    for index, row in mouse_manifest.iterrows():
        session = get_data(row.name)
        sessions.append(session)
        IDS.append(row.name)
        active.append(row.active)
    return sessions,IDS,active


def parse_stage_name_for_passive(stage_name):
    return stage_name[-7:] == "passive"

def check_duplicates():
    manifest = get_manifest() 
    if np.sum(manifest.set_index(['container_id','session_type']).index.duplicated()) > 0:
        raise Exception('Bad container')
        manifest = manifest.set_index(['container_id','stage'])
        manifest[manifest.index.duplicated(keep=False)]

def build_manifest_report():
    cache = get_cache()
    ophys_sessions = cache.get_session_table()
    ophys_experiments = cache.get_experiment_table()
    behavior_sessions = cache.get_behavior_session_table()

    # Ensure sessions are in the other tables
    session_ids = np.array(ophys_sessions.index)
    session_in_experiment_table = [any(ophys_experiments['ophys_session_id'] == x) for x in session_ids]
    session_in_bsession_table = [any(behavior_sessions['ophys_session_id'] == x) for x in session_ids]
    ophys_sessions['in_experiment_table'] = session_in_experiment_table
    ophys_sessions['in_bsession_table'] = session_in_bsession_table

    # Check Project Code
    good_code = ophys_sessions['project_code'].isin(['VisualBehavior', 'VisualBehaviorTask1B'])
    ophys_sessions['good_project_code'] = good_code

    # Check Session Type
    good_session = ophys_sessions['session_type'].isin(['OPHYS_1_images_A', 'OPHYS_3_images_A', 'OPHYS_4_images_B',
                                                        'OPHYS_5_images_B_passive', 'OPHYS_6_images_B', 'OPHYS_2_images_A_passive', 'OPHYS_1_images_B',
                                                        'OPHYS_2_images_B_passive', 'OPHYS_3_images_B', 'OPHYS_4_images_A', 'OPHYS_5_images_A_passive', 'OPHYS_6_images_A'])
    ophys_sessions['good_session'] = good_session

    # Active
    active = ophys_sessions['session_type'].isin(['OPHYS_1_images_A', 'OPHYS_3_images_A', 'OPHYS_4_images_B',
             'OPHYS_6_images_B', 'OPHYS_1_images_B', 'OPHYS_3_images_B', 'OPHYS_4_images_A',  'OPHYS_6_images_A'])

    ophys_sessions['active'] = active

    # Check Experiment Workflow state
    ophys_experiments['good_exp_workflow'] = ophys_experiments['experiment_workflow_state'] == "passed"

    # Check Container Workflow state
    ophys_experiments['good_container_qc_workflow'] = ophys_experiments['container_workflow_state'] == "container_qc"
    ophys_experiments['good_container_completed_workflow'] = ophys_experiments['container_workflow_state'].isin(['completed'])
    ophys_experiments['good_container_workflow'] = ophys_experiments['container_workflow_state'].isin(['completed','container_qc'])

    # Compile workflow state info into ophys_sessions
    ophys_experiments_good_workflow = ophys_experiments.query('good_exp_workflow')
    ophys_experiments_good_container_qc = ophys_experiments.query('good_container_qc_workflow')
    ophys_experiments_good_container_completed = ophys_experiments.query('good_container_completed_workflow')
    ophys_experiments_good_container = ophys_experiments.query('good_container_workflow')
    session_good_workflow = [any(ophys_experiments_good_workflow['ophys_session_id'] == x) for x in session_ids]
    container_qc_good_workflow = [any(ophys_experiments_good_container_qc['ophys_session_id'] == x) for x in session_ids]
    container_completed_good_workflow = [any(ophys_experiments_good_container_completed['ophys_session_id'] == x) for x in session_ids]
    container_good_workflow = [any(ophys_experiments_good_container['ophys_session_id'] == x) for x in session_ids]
    ophys_sessions['good_exp_workflow'] = session_good_workflow
    ophys_sessions['good_container_qc_workflow'] = container_qc_good_workflow
    ophys_sessions['good_container_completed_workflow'] = container_completed_good_workflow
    ophys_sessions['good_container_workflow'] = container_good_workflow

    bsids = ophys_sessions.behavior_session_id.values
    crashed =[]
    below_hit = []
    for index, bsid in enumerate(bsids):
        try:    
            fit = ps.load_fit(bsid)
        except:
            crashed.append(True)
            below_hit.append(False)
        else:
            crashed.append(False)
            below_hit.append(np.sum(fit['psydata']['hits']) < 50)
    ophys_sessions['model_crash'] = crashed
    ophys_sessions['low_hits'] = below_hit

    print_manifest_report(ophys_sessions)    
    return ophys_sessions

def print_manifest_report(ophys_sessions):   
    total_n = len(ophys_sessions)
    proj_n = len(ophys_sessions.query('good_project_code'))
    data_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session'))
    sess_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow'))
    acts_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & active'))
    modl_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & active & not model_crash'))
    hits_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & active & not model_crash & not low_hits'))
    cont_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & good_container_workflow'))
    ctqc_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & good_container_qc_workflow'))
    total_m = len(ophys_sessions.specimen_id.unique())
    proj_m = len(ophys_sessions.query('good_project_code').specimen_id.unique())
    data_m = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session').specimen_id.unique())
    sess_m = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow').specimen_id.unique())
    acts_m = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & active').specimen_id.unique())
    cont_m = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & (good_container_completed_workflow or good_container_qc_workflow)').specimen_id.unique())
    ctqc_m = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & good_container_qc_workflow').specimen_id.unique())
    acti_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & good_container_qc_workflow & active'))
    mice_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & good_container_qc_workflow & active').specimen_id.unique()) 
    modl_m = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & active & not model_crash').specimen_id.unique())
    hits_m = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & good_exp_workflow & active & not model_crash & not low_hits').specimen_id.unique())
    
    noQC_n = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & not good_exp_workflow & active & not low_hits'))
    noQC_m = len(ophys_sessions.query('good_project_code & in_bsession_table & in_experiment_table & good_session & not good_exp_workflow & active & not low_hits').specimen_id.unique())
    print("--------------------------------------")
    print(f"{total_n} sessions from {total_m} mice on lims")
    print(f" {proj_n} sessions from  {proj_m} mice with correct project code")
    print(f" {data_n} sessions from  {data_m} mice with no database errors")
    print(f" {sess_n} sessions from  {sess_m} mice with QC pass")
    print(f" {cont_n} sessions from  {cont_m} mice with container = completed or container_qc")
    print(f" {ctqc_n} sessions from  {ctqc_m} mice with container = container_qc")
    print("--------------------------------------")
    print(f" {acts_n} sessions from  {acts_m} mice with active behavior with session QC pass ")
    print(f" {acti_n} sessions from  {mice_n} mice with active behavior from full QC containers")
    print("--------------------------------------")
    print(f" {modl_n} sessions from  {modl_m} mice with model fits ")
    print(f" {hits_n} sessions from  {hits_m} mice with > 50 hits and model fit ")
    print("--------------------------------------")
    print(f" {noQC_n} sessions from  {noQC_m} mice with > 50 hits that failed session QC, potentially useful data")

    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(8,5))
    fs= 12
    starty = 1
    offset = 0.05
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_yticks([])
    ax.set_xticks([])

    ax.text(0,starty-offset*0,f"{total_n}",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*0,f"   sessions from {total_m} mice on LIMS",fontsize=fs)
    ax.text(0,starty-offset*1,"----------",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*1,f"   -----------------------------------------",fontsize=fs)
    ax.text(0,starty-offset*2,f"{total_n}",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*2,f"   sessions from {total_m} mice on lims",fontsize=fs)
    ax.text(0,starty-offset*3,f" {proj_n}",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*3,f"   sessions from  {proj_m} mice with correct project code",fontsize=fs)
    ax.text(0,starty-offset*4,f" {data_n}",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*4,f"   sessions from  {data_m} mice with no database errors",fontsize=fs)
    ax.text(0,starty-offset*5,f" {sess_n}",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*5,f"   sessions from  {sess_m} mice with QC pass",fontsize=fs)
    ax.text(0,starty-offset*6,f" {cont_n}",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*6,f"   sessions from  {cont_m} mice with container = completed or container_qc",fontsize=fs)
    ax.text(0,starty-offset*7,f" {ctqc_n}",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*7,f"   sessions from  {ctqc_m} mice with container = container_qc",fontsize=fs)
    ax.text(0,starty-offset*8,"----------",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*8,f"   -----------------------------------------",fontsize=fs)
    ax.text(0,starty-offset*9,f" {acts_n}",fontsize=fs,horizontalalignment='right');    ax.text(0,starty-offset*9,f"   sessions from  {acts_m} mice with active behavior with session QC pass ",fontsize=fs)
    ax.text(0,starty-offset*10,f" {acti_n}",fontsize=fs,horizontalalignment='right');   ax.text(0,starty-offset*10,f"   sessions from  {mice_n} mice with active behavior from full QC containers",fontsize=fs)
    ax.text(0,starty-offset*11,"----------",fontsize=fs,horizontalalignment='right');   ax.text(0,starty-offset*11,f"   -----------------------------------------",fontsize=fs)
    ax.text(0,starty-offset*12,f" {modl_n}",fontsize=fs,horizontalalignment='right');   ax.text(0,starty-offset*12,f"   sessions from  {modl_m} mice with model fits ",fontsize=fs)
    ax.text(0,starty-offset*13,f" {hits_n}",fontsize=fs,horizontalalignment='right');   ax.text(0,starty-offset*13,f"   sessions from  {hits_m} mice with > 50 hits and model fit ",fontsize=fs)
    ax.text(0,starty-offset*14,"----------",fontsize=fs,horizontalalignment='right');   ax.text(0,starty-offset*14,f"   -----------------------------------------",fontsize=fs)
    ax.text(0,starty-offset*15,f" {noQC_n}",fontsize=fs,horizontalalignment='right');   ax.text(0,starty-offset*15,f"   sessions from  {noQC_m} mice with > 50 hits that failed session QC, potentially useful data",fontsize=fs)
    plt.tight_layout()
    plt.savefig('/home/alex.piet/codebase/behavior/data/full_manifest_report.png')

