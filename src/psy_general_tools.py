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

def get_session_ids():
    '''
        Returns an array of ophys_experiment_ids for active and behavior sessions in the cache
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.ophys_experiment_id.values)
    return session_ids

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



