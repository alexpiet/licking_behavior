import numpy as np
from visual_behavior.data_access import reformat
import visual_behavior.data_access.loading as loading
from allensdk.brain_observatory.behavior.behavior_session import BehaviorSession
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc

#from visual_behavior.ophys.response_analysis import response_processing as rp
#from visual_behavior.ophys.response_analysis import utilities as ru

'''
This is a set of general purpose functions for interacting with the SDK
Alex Piet, alexpiet@gmail.com
11/5/2019
updated 01/22/2020
updated 04/07/2020
updated 03/01/2021
'''

def get_ophys_manifest():
    '''
        Build a table that contains all ophys sessions
    '''    
    manifest = loading.get_filtered_ophys_experiment_table(release_data_only=True).reset_index()
    manifest['active'] =  manifest['session_type'].isin(['OPHYS_1_images_A', 'OPHYS_3_images_A', 'OPHYS_4_images_A',
        'OPHYS_6_images_A',  'OPHYS_1_images_B', 'OPHYS_3_images_B', 'OPHYS_4_images_B', 'OPHYS_6_images_B'])
    manifest['passive'] = manifest['session_type'].isin(['OPHYS_2_images_A_passive', 'OPHYS_5_images_A_passive', 
        'OPHYS_2_images_B_passive', 'OPHYS_5_images_B_passive'])
    manifest['trained_A'] = manifest.session_type.isin(['OPHYS_1_images_A','OPHYS_2_images_A_passive',
        'OPHYS_3_images_A','OPHYS_4_images_B','OPHYS_5_images_B_passive','OPHYS_6_images_B'])
    manifest['trained_B'] = manifest.session_type.isin(['OPHYS_1_images_B','OPHYS_2_images_B_passive',
        'OPHYS_3_images_B','OPHYS_4_images_A','OPHYS_5_images_A_passive','OPHYS_6_images_A'])
    manifest = manifest.drop_duplicates(subset='ophys_session_id')
    manifest = manifest.drop(columns=['imaging_depth','location','model_outputs_available','ophys_experiment_id','experiment_workflow_state','session_name'])
    return manifest

def get_training_manifest():
    '''
        Return a table of all training/ophys sessions from mice in the march,2021 data release
        #UPDATE_REQUIRED, need to incorporate the various additional columns from the old notes below
    '''
    t = loading.get_filtered_behavior_session_table(release_data_only=True)
    t.sort_index(inplace=True)
    t = t.reset_index()
    t['active'] = [(x[0] == 'T') or (x[6] in ['0','1','3','4','6']) for x in t.session_type]

    return t  

    #t_manifest.drop(columns=['foraging_id','sex','full_genotype','reporter_line'],inplace=True)
    #t_manifest =t_manifest[~t_manifest.session_type.isnull()]
    #t_manifest['cre_line'] = [x[-1] for x in t_manifest.driver_line]
    #t_manifest['ophys'] = [x[0:5] =='OPHYS' for x in t_manifest.session_type]
    #t_manifest['stage'] = [x[1][6] if x[0] else x[1][9] for x in zip(t_manifest.ophys, t_manifest.session_type)]  
    #t_manifest['good'] = [True if not x[0] else True if x[1] == '0' else x[2] for x in zip(t_manifest.ophys,t_manifest.stage,t_manifest.index.isin(manifest.index))]
    #t_manifest = t_manifest.query('good').copy().drop(columns=['good'])
    #t_manifest['imaging'] = t_manifest.ophys & (t_manifest.stage >= "1")
    #t_manifest['session_number'] = t_manifest.groupby('donor_id').cumcount()
    #t_manifest['tmp'] = t_manifest.groupby(['donor_id','imaging']).cumcount()
    #t_manifest['pre_ophys_number'] = t_manifest.groupby(['donor_id','imaging']).cumcount(ascending=False)
    #t_manifest['pre_ophys_number'] = t_manifest['pre_ophys_number']+1
    #t_manifest.loc[t_manifest['imaging'],'pre_ophys_number'] = -t_manifest[t_manifest['imaging']]['tmp']
    #t_manifest= t_manifest.drop(columns=['tmp'])
    #t_manifest = t_manifest.query('(ophys) or (not ophys and stage > "2")')



def get_data(bsid,OPHYS=False):
    '''
        Loads data from SDK interface
        ARGS: bsid to load
        if OPHYS is true, loads data from the OPHYS api
    '''

    # Get core information
    if OPHYS:
        table   = loading.get_filtered_ophys_experiment_table(release_data_only=True).reset_index()
        oeid    = table.query('behavior_session_id == @bsid').iloc[0]['ophys_experiment_id']
        session = BehaviorOphysSession.from_lims(oeid)
    else:
        session = BehaviorSession.from_lims(bsid) 

    # Get extended stimulus presentations
    session.stimulus_presentations = reformat.add_change_each_flash(session.stimulus_presentations)
    session.stimulus_presentations = reformat.add_licks_each_flash(session.stimulus_presentations, session.licks)
    session.stimulus_presentations = reformat.add_rewards_each_flash(session.stimulus_presentations, session.rewards)
    session.stimulus_presentations['licked'] = [True if len(licks) > 0 else False for licks in session.stimulus_presentations.licks.values]
    session.stimulus_presentations = reformat.add_time_from_last_change(session.stimulus_presentations)
    session.stimulus_presentations = reformat.add_time_from_last_lick(session.stimulus_presentations, session.licks)
    session.stimulus_presentations = reformat.add_time_from_last_reward(session.stimulus_presentations, session.rewards)
    return session

def moving_mean(values, window):
    '''
        Computes the moving mean of the series in values, with a square window of width window
    '''
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm

################################# Old stuff below here, in development

## UPDATE REQUIRED, can probably remove 
def add_block_index_to_stimulus_response_df(session):
    # Both addsin place
    session.stimulus_presentations['block_index'] = session.stimulus_presentations.change.cumsum() 
    # Have to merge into flash_response_df
    session.flash_response_df = session.flash_response_df.merge(session.stimulus_presentations.reset_index()[['stimulus_presentations_id','block_index','start_time','image_name']],on='stimulus_presentations_id')

## UPDATE REQUIRED, can probably remove 
def get_stimulus_response_df(session):
    params = {
        "window_around_timepoint_seconds": [-0.5, 0.75],
        "response_window_duration_seconds": 0.75,
        "baseline_window_duration_seconds": 0.25,
        "ophys_frame_rate": 31,
    }
    session.flash_response_df = rp.stimulus_response_df(rp.stimulus_response_xr(session,response_analysis_params=params))
    add_block_index_to_stimulus_response_df(session)

## UPDATE REQUIRED, can probably remove 
def get_trial_response_df(session):
    session.trial_response_df = rp.trial_response_df(rp.trial_response_xr(session))
  
## UPDATE REQUIRED, can probably remove 
def get_stage(oeid):
    '''
        Returns the stage name as a string 
        ARGS: ophys_experiment_id
    '''
    ophys_experiments = cache.get_experiment_table()
    return ophys_experiments.loc[oeid]['session_type']

## UPDATE REQUIRED, can probably remove 
def get_intersection(list_of_ids):
    '''
        Returns the intersection of values in the list
    '''
    return reduce(np.intersect1d,tuple(list_of_ids))

## UPDATE REQUIRED, can probably remove 
def get_slc_session_ids():
    '''
        Returns an array of the behavior_session_ids from SLC mice 
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('cre_line == "Slc17a7-IRES2-Cre"').index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_vip_session_ids():
    '''
        Returns an array of the behavior_session_ids from VIP mice 
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('cre_line == "Vip-IRES-Cre"').index)
    return session_ids
   
## UPDATE REQUIRED, can probably remove 
def get_session_ids():
    '''
        Returns an array of the behavior_session_ids
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_active_ids():
    '''
        Returns an array of the behavior_session_ids from active sessions
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('active').index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_passive_ids():
    '''
        Returns an array of the behavior_session_ids from passive sessions
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('not active').index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_A_ids():
    '''
        Returns an array of the behavior_session_ids from sessions using image set A
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('image_set == "A"').index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_B_ids():
    '''
        Returns an array of the behavior_session_ids from sessions using image set B
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('image_set == "B"').index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_active_A_ids():
    '''
        Returns an array of the behavior_session_ids from active sessions using image set A
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('active & image_set == "A"').index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_active_B_ids():
    '''
        Returns an array of the behavior_session_ids from active sessions using image set B
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('active & image_set == "B"').index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_stage_ids(stage):
    '''
        Returns an array of the behavior_session_ids in stage 
    '''
    manifest = get_manifest()
    stage = str(stage)
    session_ids = np.unique(manifest.query('session_type.str[6] == @stage').index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_layer_ids(depth):
    '''
        Returns an array of the behavior_session_ids imaged at depth
    '''
    manifest = get_manifest()
    session_ids = np.unique(manifest.query('imaging_depth == @depth').index)
    return session_ids

## UPDATE REQUIRED, can probably remove 
def get_mice_ids():
    '''
        Returns an array of the donor_ids
    '''
    return get_donor_ids()

## UPDATE REQUIRED, can probably remove 
def get_donor_ids():
    '''
        Returns an array of the donor_ids
    '''
    manifest = get_manifest()
    mice_ids = np.unique(manifest.donor_id.values)
    return mice_ids

## UPDATE REQUIRED, can probably remove 
def get_mice_sessions(donor_id):
    '''
        Returns an array of the behavior_session_ids by mouse donor_id
    '''
    mouse_manifest = get_mouse_manifest(donor_id)
    return np.array(mouse_manifest.index)

## UPDATE REQUIRED, can probably remove 
def get_mouse_training_manifest(donor_id):
    '''
        Returns a dataframe containing all behavior_sessions for this donor_id
    '''
    t_manifest = get_training_manifest()
    mouse_t_manifest = t_manifest.query('donor_id == @donor_id').copy()
    return mouse_t_manifest
    
## UPDATE REQUIRED, can probably remove 
def get_mouse_manifest(donor_id):
    '''
        Returns a dataframe containing all ophys_sessions for this donor_id
    '''
    manifest = get_manifest()
    mouse_manifest =  manifest.query('donor_id ==@donor_id').copy()
    mouse_manifest = mouse_manifest.sort_values(by='date_of_acquisition')
    return mouse_manifest
    
## UPDATE REQUIRED, can probably remove 
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


