import os
import numpy as np
import pandas as pd
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc
from visual_behavior.translator.allensdk_sessions import sdk_utils

'''
This is a set of general purpose functions for interacting with the SDK
Alex Piet, alexpiet@gmail.com
11/5/2019
updated 01/22/2020

'''

OPHYS=True #if True, loads the data with BehaviorOphysSession, not BehaviorSession
MANIFEST_PATH = os.path.join("/home/alex.piet/codebase/behavior/manifest/", "manifest.json")

def get_data(bsid):
    '''
        Loads data from SDK interface
        ARGS: bsid to load
        if global OPHYS is true, loads data from the OPHYS api instead
    '''

    if OPHYS:
        session = get_data_from_oeid(sdk_utils.get_oeid_from_bsid(bsid,get_cache()))
    else:
        session = get_data_from_bsid(bsid)
    clean_session(session)
    return session

def clean_session(session):
    '''
        SDK PATCH
    '''
    sdk_utils.add_stimulus_presentations_analysis(session)

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
    #api=boa.BehaviorOphysLimsApi(experiment_id)
    #return api.get_task_parameters()['stage']

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
    # will need to split this when we get a behavior vs ophys manifest
    manifest = get_manifest()
    mice_ids = np.unique(manifest.donor_id.values)
    return mice_ids

def get_mice_sessions(donor_id):
    '''
        Returns an array of the behavior_session_ids by mouse donor_id
    '''
    manifest = get_manifest()
    mouse_manifest = manifest.query('donor_id == @donor_id')
    mouse_manifest = mouse_manifest.sort_values(by='date_of_acquisition')
    return np.array(mouse_manifest.index)

#####################################################################################

def get_mice_behavior_sessions(donor_id):
    # might want to add when I specify a behavior vs ophys manifest
    raise Exception('use get_mice_sessions')
    cache = get_cache()
    behavior_sessions = cache.get_behavior_session_table()
    return behavior_sessions.query('donor_id ==@donor_id').index.values

def get_mice_ophys_session(donor_id):
    # might want to add when I specify a behavior vs ophys manifest
    raise Exception('use get_mice_sessions')
    specimen_id = sdk_utils.get_specimen_id_from_donor_id(donor_id,get_cache())
    cache = get_cache()
    ophys_session = cache.get_session_table()
    return ophys_session.query('specimen_id == @specimen_id').index.values

def get_bsids_with_osids():
    raise Exception('outdated use get_session_ids()')
    osids =get_osids()
    bsids = [sdk_utils.get_bsid_from_osid(x,get_cache()) for x in osids]
    return bsids

def get_bsids():
    raise Exception('use get_session_ids()')
    cache = get_cache()
    behavior_sessions = cache.get_behavior_session_table()
    return np.unique(behavior_sessions.index.values)

def get_osids():
    raise Exception('use get_session_ids()')
    cache = get_cache()
    ophys_sessions = cache.get_session_table()
    return np.unique(ophys_sessions.index.values)

#####################################################################################

def get_behavior_manifest():
    raise Exception('not implemented')
    
def get_manifest(require_cell_matching=False,require_full_container=True,require_exp_pass=True,force_recompute=False):
    '''
        Returns a dataframe of all the ophys_sessions that satisfy the optional arguments 
    '''
    if force_recompute:
        return compute_manifest(require_cell_matching=require_cell_matching, require_full_container=require_full_container,require_exp_pass=require_exp_pass)
    elif 'behavior_manifest' in globals():
        return behavior_manifest
    else:
        return compute_manifest(require_cell_matching=require_cell_matching, require_full_container=require_full_container,require_exp_pass=require_exp_pass)

def compute_manifest(require_cell_matching=False,require_full_container=True,require_exp_pass=True):
    '''
        Returns a dataframe which is the list of all sessions in the current cache
    ''' 
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

    manifest = manifest.drop(columns=['in_experiment_table','in_bsession_table','good_project_code','good_session','good_exp_workflow','good_container_workflow','session_name'])
    global behavior_manifest
    behavior_manifest = manifest
    return manifest
    
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

def load_mouse(mouse, get_behavior=False):
    '''
        Takes a mouse donor_id, returns a list of all sessions objects, their IDS, and whether it was active or not. 
        if get_behavior, returns all BehaviorSessions
        no matter what, always returns the behavior_session_id for each session. 
        if global OPHYS, then forces get_behavior=False
    '''
    cache = pgt.get_cache()
    behavior_sessions = cache.get_behavior_session_table()
    mouse_manifest = behavior_sessions.query('donor_id ==@mouse') 
    # if global OPHYS, then forces get_behavior to be false
    if OPHYS:
        get_behavior=False    

    # Filter out behavior only sessions
    if not get_behavior: 
        mouse_manifest = mouse_manifest[~mouse_manifest['ophys_session_id'].isnull()]

    # filter out sessions with "NaN" session type
    mouse_manifest = mouse_manifest[~mouse_manifest['session_type'].isnull()]

    # needs active/passive
    active = []
    for dex, row in mouse_manifest.iterrows():
        active.append(not pgt.parse_stage_name_for_passive(row.session_type))
    mouse_manifest['active'] = active

    # needs acquisition date
    if False: # This is 100% accurate
        dates = []
        for dex, row in mouse_manifest.iterrows():
            print(dex)
            session = pgt.get_data(row.name)
            dates.append(session.metadata['experiment_datetime']) 
        mouse_manifest['dates']
        mouse_manifest = mouse_manifest.sort_values(by=['dates'])
    else: #This is probably close enough
        mouse_manifest = mouse_manifest.sort_values(by=['behavior_session_id'])

    # Load the sessions 
    sessions = []
    IDS = []
    active =[]
    for index, row in mouse_manifest.iterrows():
        session = pgt.get_data(row.name)
        sessions.append(session)
        IDS.append(row.name)
        active.append(row.active)
    return sessions,IDS,active


def parse_stage_name_for_passive(stage_name):
    return stage_name[-7:] == "passive"

