import numpy as np
import pandas as pd
from visual_behavior.data_access import reformat
import visual_behavior.data_access.loading as loading
from allensdk.brain_observatory.behavior.behavior_session import BehaviorSession
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_project_cache import BehaviorProjectCache as bpc

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

def get_training_manifest(non_ophys=True):
    '''
        Return a table of all training/ophys sessions from mice in the march,2021 data release        
        non_ophys, if True (default) removes sessions listed in get_ophys_manifest()
    '''
    training = loading.get_filtered_behavior_session_table(release_data_only=True)
    training.sort_index(inplace=True)
    training = training.reset_index()
    training['active'] = [(x[0] == 'T') or (x[6] in ['0','1','3','4','6']) for x in training.session_type]
    training['cre_line'] = [x[0] for x in training['driver_line']]
    training['ophys'] = [x[0:7] in ["OPHYS_1","OPHYS_2","OPHYS_3","OPHYS_4","OPHYS_5","OPHYS_6"] for x in training.session_type]
    training['pre_ophys_number'] = training.groupby(['donor_id','ophys']).cumcount(ascending=False)
    training['training_number'] = training.groupby(['donor_id']).cumcount(ascending=True)+1
    training['tmp'] = training.groupby(['donor_id','ophys']).cumcount()
    training['pre_ophys_number'] = training['pre_ophys_number']+1
    training.loc[training['ophys'],'pre_ophys_number'] = -training[training['ophys']]['tmp']
    training= training.drop(columns=['tmp'])

    if non_ophys:
        manifest = get_ophys_manifest()
        training = training[~training.behavior_session_id.isin(manifest.behavior_session_id)] 
    return training 
    # include trained_A, trained_B? maybe should generalize to trained = {'A','B','G','H'}


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
 
    training_0_1 = session.metadata['session_type'] in ["TRAINING_1_gratings","TRAINING_0_gratings_autorewards_15min"]
    if training_0_1:
        session = build_pseudo_stimulus_presentations(session)

    # Get extended stimulus presentations
    session.stimulus_presentations = reformat.add_change_each_flash(session.stimulus_presentations)
    if training_0_1:
        session.stimulus_presentations = training_add_licks_each_flash(session.stimulus_presentations, session.licks)
        session.stimulus_presentations = training_add_rewards_each_flash(session.stimulus_presentations, session.rewards)
    else:
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
   
def get_session_ids():
    '''
        Returns an array of the behavior_session_ids
    '''
    manifest = get_ophys_manifest()
    session_ids = np.unique(manifest.behavior_session_id)
    return session_ids

def get_active_ids():
    '''
        Returns an array of the behavior_session_ids from active sessions
    '''
    manifest = get_ophys_manifest()
    session_ids = np.unique(manifest.query('active').behavior_session_id)
    return session_ids

def get_passive_ids():
    '''
        Returns an array of the behavior_session_ids from passive sessions
    '''
    manifest = get_ophys_manifest()
    session_ids = np.unique(manifest.query('not active').behavior_session_id)
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
def get_mice_ids(OPHYS=True):
    '''
        Returns an array of the donor_ids
    '''
    if OPHYS:
        manifest = get_ophys_manifest()
    else:
        manifest = get_training_manifest()
    
    return manifest.donor_id.unique()

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

def build_pseudo_stimulus_presentations(session):
    '''
        For Training 0/1 the stimulus was not flashes but presented serially. This
        function builds a pseudo table of stimuli by breaking up the continuously
        presented stimuli into repeated stimuli. This is just to make the behavior model
        fit. 
    '''
    # Store the original
    session.stimulus_presentations_sdk = session.stimulus_presentations.copy()

    # Get the basic data frame by iterating start times
    session.stimulus_presentations = pd.DataFrame()
    start_times = []
    image_index = []
    image_name =[]
    for index, row in session.stimulus_presentations_sdk.iterrows():
        new_images = list(np.arange(row['start_time'],row['stop_time'],0.75))
        start_times = start_times+ new_images
        image_index = image_index + [row['image_index']]*len(new_images) 
        image_name = image_name + [row['image_name']]*len(new_images) 
    session.stimulus_presentations['start_time'] = start_times
    session.stimulus_presentations['image_index'] = image_index
    session.stimulus_presentations['image_name'] = image_name

    # Filter out very short stimuli which happen because the stimulus duration was not
    # constrainted to be a multiple of 750ms
    session.stimulus_presentations['duration'] = session.stimulus_presentations.shift(-1)['start_time']-session.stimulus_presentations['start_time']
    session.stimulus_presentations = session.stimulus_presentations.query('duration > .25').copy()
    session.stimulus_presentations['duration'] = session.stimulus_presentations.shift(-1)['start_time']-session.stimulus_presentations['start_time']


    # Add other columns
    session.stimulus_presentations['omitted'] = False
    session.stimulus_presentations['stop_time'] = session.stimulus_presentations['duration']+session.stimulus_presentations['start_time']
    session.stimulus_presentations['image_set'] = session.stimulus_presentations_sdk.iloc[0]['image_set']

    return session

def training_add_licks_each_flash(stimulus_presentations, licks):
    lick_times = licks['timestamps'].values
    licks_each_flash = stimulus_presentations.apply(
        lambda row: lick_times[((lick_times > row["start_time"]) & (lick_times < row["stop_time"]))],
        axis=1)
    stimulus_presentations['licks'] = licks_each_flash
    return stimulus_presentations

def training_add_rewards_each_flash(stimulus_presentations,rewards):
    reward_times = rewards['timestamps'].values
    rewards_each_flash = stimulus_presentations.apply(
        lambda row: reward_times[((reward_times > row["start_time"]) & (reward_times < row["stop_time"]))],
        axis=1,
    )
    stimulus_presentations['rewards'] = rewards_each_flash
    return stimulus_presentations


