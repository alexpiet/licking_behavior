import numpy as np
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc
import pandas as pd
from allensdk.internal.api import behavior_ophys_api as boa

'''
This is a set of general purpose functions for interacting with the SDK
Alex Piet, alexpiet@gmail.com
11/5/2019
updated 01/22/2020

'''

MANIFEST_PATH = os.path.join("/home/alex.piet/codebase/behavior/manifest/", "manifest.json")

### Functions to update
##  get_active_A_ids()
##  get_active_B_ids()
##  get_layer_ids()
##  get_stage_ids()
##  get_active_ids()
##  get_passive_ids()
##  get_A_ids()
##  get_B_ids()
##  get_slc_session_ids()
##  get_vip_session_ids()
##  get_mice_ids()
##  get_ophys_ids()


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
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm
 
def get_stage(experiment_id):
    '''
       Returns the stage name as a string 
    '''
    api=boa.BehaviorOphysLimsApi(experiment_id)
    return api.get_task_parameters()['stage']

def get_intersection(list_of_ids):
    return reduce(np.intersect1d,tuple(list_of_ids))

def get_slc_session_ids():
    raise Exception('outdated')
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['cre_line'] == 'Slc17a7-IRES2-Cre'].ophys_experiment_id.values)
    return session_ids

def get_vip_session_ids():
    raise Exception('outdated')
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['cre_line'] == 'Vip-IRES-Cre'].ophys_experiment_id.values)
    return session_ids
   
def get_session_ids():
    raise Exception('outdated')
    manifest = get_manifest()
    session_ids = np.unique(manifest.ophys_experiment_id.values)
    return session_ids

def get_mice_ids():
    raise Exception('outdated')
    manifest = get_manifest()
    mice_ids = np.unique(manifest.animal_name.values)
    return mice_ids

def get_mice_sessions(mouse_id):
    '''
    
        NEED
    '''
    raise Exception('outdated')
    manifest = get_manifest()
    mouse_manifest = manifest[manifest['animal_name'] == int(mouse_id)]
    mouse_manifest = mouse_manifest.sort_values(by='date_of_acquisition')
    return mouse_manifest.ophys_experiment_id.values


def get_active_ids():
    '''
        Returns an array of ophys_experiment_ids for active behavior sessions
    '''
    raise Exception('outdated')
    manifest = get_manifest()
    session_ids = np.unique(manifest[~manifest['passive_session']].ophys_experiment_id.values)
    return session_ids

def get_A_ids():
    raise Exception('outdated')
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['image_set'] == 'A'].ophys_experiment_id.values)
    return session_ids

def get_B_ids():
    raise Exception('outdated')
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['image_set'] == 'B'].ophys_experiment_id.values)
    return session_ids

def get_passive_ids():
    raise Exception('outdated')
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['passive_session']].ophys_experiment_id.values)
    return session_ids

def get_active_A_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[(~manifest['passive_session']) &(manifest['image_set']=='A')].ophys_experiment_id.values)
    return session_ids

def get_active_B_ids():
    manifest = get_manifest()
    session_ids = np.unique(manifest[(~manifest['passive_session']) &(manifest['image_set']=='B')].ophys_experiment_id.values)
    return session_ids

def get_stage_ids(stage):
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['stage_name'].str[6] == str(stage)].ophys_experiment_id.values)
    return session_ids

def get_layer_ids(depth):
    raise Exception('outdated')
    manifest = get_manifest()
    session_ids = np.unique(manifest[manifest['imaging_depth'] == depth].ophys_experiment_id.values)
    return session_ids

def get_mice_donor_ids():
    cache = get_cache()
    behavior_sessions = cache.get_behavior_session_table()    
    return behavior_sessions['donor_id'].unique()

def get_mice_donor_ids_with_ophys():
    specimen_ids = get_mice_specimen_ids()
    donor_ids = [sdk_utils.get_donor_id_from_specimen_id(x,get_cache()) for x in specimen_ids]
    return donor_ids

def get_mice_specimen_ids():
    cache = get_cache()
    ophys_sessions = cache.get_session_table()    
    return ophys_sessions['specimen_id'].unique()

def get_mice_behavior_sessions(donor_id):
    cache = get_cache()
    behavior_sessions = cache.get_behavior_session_table()
    return behavior_sessions.query('donor_id ==@donor_id').index.values

def get_mice_ophys_session(donor_id):
    specimen_id = sdk_utils.get_specimen_id_from_donor_id(donor_id,get_cache())
    cache = get_cache()
    ophys_session = cache.get_session_table()
    return ophys_session.query('specimen_id == @specimen_id').index.values

def get_manifest():
    '''
        Returns a dataframe which is the list of all sessions in the current cache
    '''
    cache = get_cache()
    manifest = cache.experiment_table
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

def get_bsids():
    '''
        Return a list of all bsids
    '''
    cache = get_cache()
    behavior_sessions = cache.get_behavior_session_table()
    return np.unique(behavior_sessions.index.values)

def get_osids():
    '''
        Return a list of all osids
    '''
    cache = get_cache()
    ophys_sessions = cache.get_session_table()
    return np.unique(ophys_sessions.index.values)

def get_bsids_with_osids():
    osids =get_osids()
    bsids = [sdk_utils.get_bsid_from_osid(x,get_cache()) for x in osids]
    return bsids

def load_mouse(mouse, get_behavior=False,OPHYS=True):
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

