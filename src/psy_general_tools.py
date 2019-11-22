import numpy as np
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc
import pandas as pd

'''
This is a set of general purpose functions for interacting with the SDK
Alex Piet, alexpiet@gmail.com
11/5/2019

'''


def get_active_ids():
    '''
        Returns an array of ophys_experiment_ids for active behavior sessions
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest[~manifest['passive_session']].ophys_experiment_id.values)
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

def get_session_ids():
    '''
        Returns an array of ophys_experiment_ids for active and behavior sessions in the cache
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.ophys_experiment_id.values)
    return session_ids

def get_mice_ids():
    manifest = get_manifest()
    mice_ids = np.unique(manifest.animal_name.values)
    return mice_ids 

def get_manifest():
    '''
        Returns a dataframe which is the list of all sessions in the current cache
    '''
    cache = get_cache()
    manifest = cache.experiment_table
    return manifest

def get_cache():
    '''
        Returns the current cache of sessions
    '''
    cache_json = '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/cache_20190813'
    cache = bpc.BehaviorProjectCache(cache_json)
    return cache

def get_data(experiment_id,stage="",load_dir = r'/allen/aibs/technology/nicholasc/behavior_ophys'):
    '''
        Loads data from SDK interface
        ARGS: experiment_id to load
        Returns the SDK object
    '''
    cache = get_cache()
    session = cache.get_session(experiment_id)
    return session


def load_mouse(mouse,get_ophys=True, get_behavior=False):
    '''
        Takes a mouse donor_id, and filters the sessions in Nick's database, and returns a list of session objects. Optional arguments filter what types of sessions are returned    
    '''
    #print("This function may be out dated, try using get_mice_sessions()")
    manifest = get_manifest()
    mouse_manifest = manifest[manifest['animal_name'] == int(mouse)]
    mouse_manifest = mouse_manifest.sort_values(by='date_of_acquisition')
 
    #vb_sessions = pd.read_hdf('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/vb_sessions.h5', key='df')
    #vb_sessions_good = vb_sessions[vb_sessions['stage_name'] != 'Load error']
    #mouse_session =  vb_sessions_good[vb_sessions_good['donor_id'] == mouse]  
 
    sessions = []
    IDS = []
    active =[]
    for index, row in mouse_manifest.iterrows():
        session, session_id = load_session(row,get_ophys=get_ophys, get_behavior=get_behavior)
        if not (type(session) == type(None)): 
            sessions.append(session)
            IDS.append(session_id)
            active.append(not parse_stage_name_for_passive(row.stage_name))
    return sessions,IDS,active

def parse_stage_name_for_passive(stage_name):
    return stage_name[-1] == "e"

def load_session(row,get_ophys=True, get_behavior=False):
    '''
        Takes in a row of Nick's database of sessions and loads a session either via the ophys interface or behavior interface. Two optional arguments toggle what types of data are returned 
    '''
    #print("This function may be out dated, try using get_data()")
    if pd.isnull(row['ophys_experiment_id']):
        session_id = 'behavior_{}'.format(int(row['behavior_session_id']))
        if get_behavior:
            # this will crash because we haven't supported bs yet
            api = bla.BehaviorLimsApi(int(row['behavior_session_id']))
            session = bs.BehaviorSession(api)
        else:
            session = None
    else:
        session_id = 'ophys_{}'.format(int(row['ophys_experiment_id']))
        if get_ophys:
            session = get_data(int(row['ophys_experiment_id']))
            session.metadata['stage'] = row.stage_name
        else:
            session = None
    return session, session_id


def get_mice_sessions(mouse_id):
    manifest = get_manifest()
    mouse_manifest = manifest[manifest['animal_name'] == int(mouse_id)]
    mouse_manifest = mouse_manifest.sort_values(by='date_of_acquisition')
    return mouse_manifest.ophys_experiment_id.values
