import json
import numpy as np
import pandas as pd
from visual_behavior.data_access import reformat 
import visual_behavior.data_access.loading as loading
import visual_behavior.data_access.utilities as utilities
from allensdk.brain_observatory.behavior.behavior_session import BehaviorSession
from allensdk.brain_observatory.behavior.behavior_ophys_session import \
    BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache

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
    2:957032492,
    3:1041977344,
    4:873695653,
    5:824894721,
    6:907053876,
    7:894083470,
    8:820124540,
    9:1066967257,
    10:884808160
    } 
    return test_ids[num]

def get_directory(version,verbose=False,subdirectory=None,group=None):
    root_directory  = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/'
    if subdirectory =='fits':
        subdir = 'session_fits/'
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
    np_table['EPHYS_session_type'] = [
        x.startswith('EPHYS') for x in np_table['session_type']]  
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
    experiments_table = experiments_table[
        (experiments_table.project_code!="VisualBehaviorMultiscope4areasx2d")&\
        (experiments_table.reporter_line!="Ai94(TITL-GCaMP6s)")].reset_index()
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
        session_table = session_table[\
            session_table.reporter_line!="Ai94(TITL-GCaMP6s)"\
            ].reset_index()   
    else:
        session_table = session_table[
            (session_table.project_code!="VisualBehaviorMultiscope4areasx2d")&\
            (session_table.reporter_line!="Ai94(TITL-GCaMP6s)")]\
            .reset_index()
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
    Add a column called 'n_relative_to_first_novel' that indicates the session 
    number relative to the first novel session for each experiment in a container.
    If a container does not have a first novel session, the value of 
    n_relative_to_novel for all experiments in the container is NaN.
    Input df must have column 'experience_level' and 'date'
    Input df is typically ophys_experiment_table
    """
    # add simplified string date column for accurate sorting

    df = df.sort_values(by=['mouse_id', 'date'])  # must sort for ordering to be accurate
    numbers = df.groupby('mouse_id').apply(utilities.get_n_relative_to_first_novel)
    df['n_relative_to_first_novel'] = np.nan
    for mouse_id in df.mouse_id.unique():
        indices = df[df.mouse_id == mouse_id].index.values
        df.loc[indices, 'n_relative_to_first_novel'] = \
            list(numbers.loc[mouse_id].n_relative_to_first_novel)
    return df

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
        table =\
            loading.get_filtered_ophys_experiment_table(release_data_only=True)\
            .reset_index()
        oeid = \
            table.query('behavior_session_id == @bsid').iloc[0]['ophys_experiment_id']
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

def moving_mean(values, window,mode='valid'):
    '''
        Computes the moving mean of the series in values, with a square window 
        of width window
    '''
    weights = np.repeat(1.0, window)/window
    mm = np.convolve(values, weights, mode)
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
        'bias':'licking bias',
        'omissions':'omission',
        'omissions0':'omission',
        'Omissions':'omission',
        'Omissions1':'post omission',
        'omissions1':'post omission',
        'task0':'visual',
        'Task0':'visual',
        'timing1D':'timing',
        'Full-Task0':'full model',
        'dropout_task0':'Visual Dropout',    
        'dropout_timing1D':'Timing Dropout', 
        'dropout_omissions':'Omission Dropout',
        'dropout_omissions1':'Post Omission Dropout',
        'Sst-IRES-Cre' :'Sst Inhibitory',
        'Vip-IRES-Cre' :'Vip Inhibitory',
        'Slc17a7-IRES2-Cre' :'Excitatory',
        'strategy_dropout_index': 'strategy index',
        'num_hits':'rewards/session',
        'num_miss':'misses/session',
        'num_image_false_alarm':'false alarms/session',
        'num_post_omission_licks':'post omission licks/session',
        'num_omission_licks':'omission licks/session',
        'post_reward':'previous bout rewarded',
        'not post_reward':'previous bout unrewarded',
        'timing1':'1',
        'timing2':'2',
        'timing3':'3',
        'timing4':'4',
        'timing5':'5',
        'timing6':'6',
        'timing7':'7',
        'timing8':'8',
        'timing9':'9',
        'timing10':'10',    
        'not visual_strategy_session':'timing sessions',
        'visual_strategy_session':'visual sessions',
        'visual_only_dropout_index':'visual index',
        'timing_only_dropout_index':'timing index',
        'lick_hit_fraction_rate':'lick hit fraction',
        'session_roc':'dynamic model (AUC)',
        'miss':'misses',
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
        'Familiar':'familiar',
        'Novel 1':'novel',  
        'Novel >1':'novel+'}

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
    elif version in [22]:
        strategies=['bias','task0','timing1D']
    else:
        raise Exception('Unknown model version')
    return strategies


def get_engagement_threshold():
    '''
        Definition for engagement in units of rewards/sec
    '''
    return 1/120

    
def get_bout_threshold():
    '''
        The minimum time between licks to segment licks into bouts
        700ms, or .7s
    '''
    return .7



