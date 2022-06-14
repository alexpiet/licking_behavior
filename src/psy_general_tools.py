import json
import numpy as np
import pandas as pd
from visual_behavior.data_access import reformat 
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
from allensdk.brain_observatory.behavior.behavior_session import BehaviorSession
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorOphysProjectCache

'''
This is a set of general purpose functions for interacting with the SDK
Alex Piet, alexpiet@gmail.com
11/5/2019
updated 01/22/2020
updated 04/07/2020
updated 03/01/2021
updated 02/11/2022
'''
def get_debugging_id(num=1):
    '''
        Just a series of behavior_session_ids used as fixed debugging examples
    '''
    test_ids = {
    1:951520319,
    2:957032492
    } 
    return test_ids[num]

def get_directory(version,verbose=False,subdirectory=None,group=None):
    root_directory  = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/'
    if subdirectory =='fits':
        subdir = 'session_fits/'
    elif subdirectory == 'clusters':
        subdir = 'session_clusters/'
    elif subdirectory == "strategy_df":
        subdir = 'session_strategy_df/'
    elif subdirectory == "licks_df":
        subdir = 'session_licks_df/'
    elif subdirectory == 'summary':
        subdir = 'summary_data/'
    elif subdirectory == 'figures':
        subdir = 'figures_summary/'
    elif subdirectory == 'session_figures':
        subdir = 'figures_sessions/'
    elif subdirectory == 'training_figures':
        subdir = 'figures_training/'
    elif subdirectory is None:
        subdir = ''
    else:
        raise Exception('Unkown subdirectory')
    if (group is not None) and (group != ""):
        subdir += group+'/'

    directory = root_directory+'psy_fits_v'+str(version)+'/'+subdir
    return directory

def get_np_session_table():
    filename = '/allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/metadata_220429/behavior_sessions.csv'
    np_table= pd.read_csv(filename)
    np_table['EPHYS_session_type'] = [x.startswith('EPHYS') for x in np_table['session_type']]  
    np_table['EPHYS_rig'] = [x in ["NP.1", "NP.0"] for x in np_table['equipment_name']]
    np_table['EPHYS'] = np_table['EPHYS_rig'] & np_table['EPHYS_session_type'] 
    return np_table

def get_ophys_experiment_table():
    '''
        Returns a table of all the ophys experiments in the platform paper cache
    '''
    raise Exception('You probably want get_ophys_session_table')
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache'
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    experiments_table = cache.get_ophys_experiment_table()
    experiments_table = experiments_table[(experiments_table.project_code!="VisualBehaviorMultiscope4areasx2d")&(experiments_table.reporter_line!="Ai94(TITL-GCaMP6s)")].reset_index()
    return experiments_table
    
def get_ophys_session_table(include_4x2=False):
    '''
        Returns a table of all the ophys sessions in the platform paper cache
        include_4x2 (bool), removes 4x2 Multiscope data
    '''
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache'
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    session_table = cache.get_ophys_session_table()
    if include_4x2:
        session_table = session_table[session_table.reporter_line!="Ai94(TITL-GCaMP6s)"].reset_index()   
    else:
        session_table = session_table[(session_table.project_code!="VisualBehaviorMultiscope4areasx2d")&(session_table.reporter_line!="Ai94(TITL-GCaMP6s)")].reset_index()
    return session_table

def get_ophys_manifest(include_4x2=False):
    '''
        Build a table that contains all active ophys sessions
        Adds columns for whether the mouse trained on image set A or B
    '''    
    manifest = get_ophys_session_table(include_4x2=include_4x2)
    manifest['active'] = manifest['session_type'].isin([
        'OPHYS_1_images_A', 'OPHYS_3_images_A', 'OPHYS_4_images_A', 
        'OPHYS_6_images_A',  'OPHYS_1_images_B', 'OPHYS_3_images_B', 
        'OPHYS_4_images_B', 'OPHYS_6_images_B','OPHYS_1_images_G',
        'OPHYS_3_images_G','OPHYS_4_images_G','OPHYS_6_images_G',
        'OPHYS_1_images_H','OPHYS_3_images_H','OPHYS_4_images_H',
        'OPHYS_6_images_H'])
    manifest['trained_A'] = manifest.session_type.isin([
        'OPHYS_1_images_A','OPHYS_2_images_A_passive','OPHYS_3_images_A',
        'OPHYS_4_images_B','OPHYS_5_images_B_passive','OPHYS_6_images_B'])
    manifest['trained_B'] = manifest.session_type.isin([
        'OPHYS_1_images_B','OPHYS_2_images_B_passive','OPHYS_3_images_B',
        'OPHYS_4_images_A','OPHYS_5_images_A_passive','OPHYS_6_images_A'])
    manifest['novel_session'] = [x in [4,5,6] for x in manifest.session_number]
    manifest = utilities.add_experience_level_to_experiment_table(manifest)
    manifest = add_detailed_experience_level(manifest)
    manifest = manifest.sort_index()
    manifest = manifest.query('active')
    return manifest


def add_detailed_experience_level(manifest):
    '''
        Replacement for messy functions in visual_behavior.data_access.utilities
    '''
    manifest = utilities.add_date_string(manifest)  
    manifest = add_n_relative_to_first_novel(manifest) 
    manifest = utilities.add_last_familiar_column(manifest)
    manifest = utilities.add_second_novel_column(manifest)
    manifest['strict_experience'] = (manifest['experience_level'] == 'Novel 1') |\
        (manifest['last_familiar']) | (manifest['second_novel'])
    return manifest


def add_n_relative_to_first_novel(df):
    """
    Add a column called 'n_relative_to_first_novel' that indicates the session number relative to the first novel session for each experiment in a container.
    If a container does not have a first novel session, the value of n_relative_to_novel for all experiments in the container is NaN.
    Input df must have column 'experience_level' and 'date'
    Input df is typically ophys_experiment_table
    """
    # add simplified string date column for accurate sorting

    df = df.sort_values(by=['mouse_id', 'date'])  # must sort for ordering to be accurate
    numbers = df.groupby('mouse_id').apply(utilities.get_n_relative_to_first_novel)
    df['n_relative_to_first_novel'] = np.nan
    for mouse_id in df.mouse_id.unique():
        indices = df[df.mouse_id == mouse_id].index.values
        df.loc[indices, 'n_relative_to_first_novel'] = list(numbers.loc[mouse_id].n_relative_to_first_novel)
    return df


def get_training_manifest(non_ophys=True): #TODO, Issue #92
    '''
        Return a table of all training/ophys sessions from mice in the march,2021 data release        
        non_ophys, if True (default) removes sessions listed in get_ophys_manifest()
    '''
    raise Exception('Need to update')
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

def load_version_parameters(VERSION):
    json_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v'+str(VERSION)+'/summary_data/behavior_model_params.json'
    with open(json_path,'r') as json_file:
        format_options = json.load(json_file)
    return format_options

def get_data(bsid,OPHYS=False, NP=False):
    '''
        Loads data from SDK interface
        ARGS: bsid to load
        if OPHYS is true, loads data from the OPHYS api
    '''
    assert not (OPHYS and NP), "Cannot have both OPHYS and NP Flags"

    # Get SDK session object
    print('Loading SDK object')
    if OPHYS:
        # pick an associated experiment_id
        table   = loading.get_filtered_ophys_experiment_table(release_data_only=True).reset_index()
        oeid    = table.query('behavior_session_id == @bsid').iloc[0]['ophys_experiment_id']
        session = BehaviorOphysSession.from_lims(oeid)
    elif NP:
        raise Exception('Not implemented')
        # from allensdk.brain_observatory.ecephys.behavior_ecephys_session import VBNBehaviorSession
        # directory as of May 5th, 2022. Should update once data is released
        # nwb_dir = '/allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/nwbs_220429/'
        # somehow get esid
        # filepath = nwb_dir + 'ecephys_session_'+esid+'.nwb'
        # session = VBNBehaviorSession.from_nwb_path(filepath) 
        # gives a KeyError
    else:
        session = BehaviorSession.from_lims(bsid)

    print('Checking for early omission')
    while session.stimulus_presentations.iloc[0]['omitted'] == True:
        print('Removing early omission')
        session.stimulus_presentations.drop(index=[0],inplace=True)
 
    print('Adding stimulus annotations')
    if session.metadata['session_type'] in \
        ["TRAINING_1_gratings","TRAINING_0_gratings_autorewards_15min"]: 
        raise Exception('Need to update')
        session = build_pseudo_stimulus_presentations(session)
        session.stimulus_presentations = training_add_licks_each_image(\
            session.stimulus_presentations, session.licks)
        session.stimulus_presentations = training_add_rewards_each_image(\
            session.stimulus_presentations, session.rewards)
    else:
        # Get extended stimulus presentations
        reformat.add_licks_each_flash(session.stimulus_presentations, session.licks) 
        reformat.add_rewards_each_flash(session.stimulus_presentations, session.rewards)

    session.stimulus_presentations['licked'] = [True if len(licks) > 0 else False \
        for licks in session.stimulus_presentations.licks.values]
    reformat.add_time_from_last_change(session.stimulus_presentations) 
    reformat.add_time_from_last_lick(session.stimulus_presentations, session.licks)
    reformat.add_time_from_last_reward(session.stimulus_presentations, session.rewards)
    return session

def moving_mean(values, window):
    '''
        Computes the moving mean of the series in values, with a square window of width window
    '''
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, 'valid')
    return mm

def get_clean_rate(vector, length=4800):
    if len(vector) >= length:
        return vector[0:length]
    else:
        return np.concatenate([vector, [np.nan]*(length-len(vector))])

def get_clean_string(strings):
    '''
        Return a cleaned up list of weights suitable for plotting labels
    '''
    string_dict = {
        'bias':'Bias',
        'omissions':'Omission',
        'omissions0':'Omission',
        'Omissions':'Omission',
        'Omissions1':'Post Omission',
        'omissions1':'Post Omission',
        'task0':'Visual',
        'Task0':'Visual',
        'timing1D':'Timing',
        'Full-Task0':'Full Model',
        'dropout_task0':'Visual Dropout',    
        'dropout_timing1D':'Timing Dropout', 
        'dropout_omissions':'Omission Dropout',
        'dropout_omissions1':'Post Omission Dropout',
        'Sst-IRES-Cre' :'Sst Inhibitory',
        'Vip-IRES-Cre' :'Vip Inhibitory',
        'Slc17a7-IRES2-Cre' :'Excitatory'
        }

    clean_strings = []
    for w in strings:
        if w in string_dict.keys():
            clean_strings.append(string_dict[w])
        else:
            clean_strings.append(str(w).replace('_',' '))
    return clean_strings

def get_clean_session_names(strings):
    string_dict = {
        1:'F1',
        2:'F2',
        3:'F3',
        4:'N1',
        5:'N2',
        6:'N3',
        '1':'F1',
        '2':'F2',
        '3':'F3',
        '4':'N1',
        '5':'N2',
        '6':'N3',
        'Familiar':'Familiar',
        'Novel 1':'Novel 1'}

    clean_strings = []
    for w in strings:
        if w in string_dict.keys():
            clean_strings.append(string_dict[w])
        else:
            clean_strings.append(str(w).replace('_',' '))
    return clean_strings


def get_strategy_list(version):
    '''
        Returns a sorted list of the strategies in model <version>

        Raises an exception if the model version is not recognized. 
    '''
    if version in [20,21]:
        strategies=['bias','omissions','omissions1','task0','timing1D']
    else:
        raise Exception('Unknown model version')
    return strategies

def get_engagement_threshold():
    '''
        Definition for engagement in units of rewards/sec
        1 reward every 90 seconds
    '''
    # TODO, Issue #213
    return 1/180
    #return 1/90

def get_bout_threshold():
    '''
        The minimum time between licks to segment licks into bouts
        700ms, or .7s
    '''
    return .7


## Training functions below here, in development
################################# 

def build_pseudo_stimulus_presentations(session):#TODO, Issue #92
    '''
        For Training 0/1 the stimulus was not images but presented serially. This
        function builds a pseudo table of stimuli by breaking up the continuously
        presented stimuli into repeated stimuli. This is just to make the behavior model
        fit. 
    '''
    raise Exception('Need to update')
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
    session.stimulus_presentations = session.stimulus_presentations.query('duration > .25').copy().reset_index()
    session.stimulus_presentations['duration'] = session.stimulus_presentations.shift(-1)['start_time']-session.stimulus_presentations['start_time']


    # Add other columns
    session.stimulus_presentations['omitted'] = False
    session.stimulus_presentations['stop_time'] = session.stimulus_presentations['duration']+session.stimulus_presentations['start_time']
    session.stimulus_presentations['image_set'] = session.stimulus_presentations_sdk.iloc[0]['image_set']

    return session

def training_add_licks_each_image(stimulus_presentations, licks):#TODO, Issue #92
    raise Exception('Need to update')
    lick_times = licks['timestamps'].values
    licks_each_image = stimulus_presentations.apply(
        lambda row: lick_times[((lick_times > row["start_time"]) & (lick_times < row["stop_time"]))],
        axis=1)
    stimulus_presentations['licks'] = licks_each_image
    return stimulus_presentations

def training_add_rewards_each_image(stimulus_presentations,rewards):#TODO, Issue #92
    raise Exception('Need to update')
    reward_times = rewards['timestamps'].values
    rewards_each_image = stimulus_presentations.apply(
        lambda row: reward_times[((reward_times > row["start_time"]) & (reward_times < row["stop_time"]))],
        axis=1,
    )
    stimulus_presentations['rewards'] = rewards_each_image
    return stimulus_presentations


