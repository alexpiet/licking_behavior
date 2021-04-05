import psy_general_tools as pgt
import psy_metrics_tools as pm
import psy_tools as ps
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

OUTPUT_DIR = '/home/alex.piet/codebase/behavior/model_output/'

def build_id_fit_list(VERSION):
    '''
        Saves out two text files with lists of all behavior_session_ids for ophys and training sessions in the manifest
        Only includes active sessions
    '''
    # Get manifest
    manifest = pgt.get_ophys_manifest()
    training = pgt.get_training_manifest()
 
    # Set filenames
    fname = '/home/alex.piet/codebase/behavior/licking_behavior/scripts/psy_ids_v'+str(VERSION)+'.txt'
    ftname ='/home/alex.piet/codebase/behavior/licking_behavior/scripts/psy_training_ids_v'+str(VERSION)+'.txt'

    # Filter and save
    np.savetxt(fname,  manifest.query('active')['behavior_session_id'].values)
    np.savetxt(ftname, training.query('active')['behavior_session_id'].values)

    # Make appropriate folders 
    root_directory  = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/'
    directory = root_directory+'psy_fits_v'+str(VERSION)
    if not os.path.isdir(directory):
        os.mkdir(directory)
        os.mkdir(directory+'/figures_summary')
        os.mkdir(directory+'/figures_sessions')
        os.mkdir(directory+'/psytrack_logs')
    else:
        print('directory already exists')

def get_ophys_summary_table(version):
    model_dir = ps.get_directory(version)
    return pd.read_pickle(model_dir+'_summary_table.pkl')

def build_summary_table(version):
    ''' 
        Saves out the model manifest as a csv file 
    '''
    model_manifest = ps.build_model_manifest(version=version,container_in_order=False)
    model_manifest.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D'],inplace=True) #this are in time units of bouts, we need time-aligned weights
    model_manifest = add_time_aligned_session_info(model_manifest)
    model_manifest = build_strategy_matched_subset(model_manifest)
    model_dir = ps.get_directory(version) 
    model_manifest.to_pickle(model_dir+'_summary_table.pkl')
    model_manifest.to_pickle(OUTPUT_DIR+'_summary_table.pkl')

def add_time_aligned_session_info(manifest):
    ## TODO things to add
    #  hit,miss
    weight_columns = {'bias','task0','omissions','omissions1','timing1D'}
    for column in weight_columns:
        manifest['weight_'+column] = [[]]*len(manifest)
    columns = {'lick_bout_rate','reward_rate','engaged','lick_hit_fraction','hit','miss','FA','CR'}
    for column in columns:
        manifest[column] = [[]]*len(manifest)      
    crash = 0
    for index, row in tqdm(manifest.iterrows(),total=manifest.shape[0]):
        try:
            session_df = pd.read_csv(OUTPUT_DIR+str(row.behavior_session_id)+'.csv')
            session_df['hit'] = session_df['rewarded']
            session_df['miss'] = session_df['change'] & ~session_df['rewarded']
            session_df['FA'] = session_df['lick_bout_start'] & session_df['rewarded']
            session_df['CR'] = ~session_df['lick_bout_start'] & ~session_df['change']
            if 'hit_fraction' in session_df:
                session_df['lick_hit_fraction'] = session_df['hit_fraction']
            for column in weight_columns:
                manifest.at[index, 'weight_'+column] = pgt.get_clean_rate(session_df[column].values)
            for column in columns:
                manifest.at[index, column] = pgt.get_clean_rate(session_df[column].values)
        except Exception as e:
            crash +=1
            print(e)
            for column in weight_columns:
                manifest.at[index, 'weight_'+column] = np.array([np.nan]*4800)
            for column in columns:
                manifest.at[index, column] = np.array([np.nan]*4800) 
    if crash > 0:
        print(str(crash) + ' sessions crashed, consider running build_all_session_outputs')
    return manifest 

def build_strategy_matched_subset(manifest):
    manifest['strategy_matched'] = True
    manifest.loc[(manifest['cre_line'] == "Slc17a7-IRES2-Cre")&(manifest['visual_only_dropout_index'] < -10),'strategy_matched'] = False
    manifest.loc[(manifest['cre_line'] == "Vip-IRES-Cre")&(manifest['timing_only_dropout_index'] < -15)&(manifest['timing_only_dropout_index'] > -20),'strategy_matched'] = False
    return manifest

def get_training_summary_table(version):
    model_dir = ps.get_directory(version)
    return pd.read_csv(model_dir+'_training_summary_table.csv')

def build_training_summary_table(version):
    ''' 
        Saves out the model manifest as a csv file 
    '''
    model_manifest = ps.build_model_training_manifest(version)
    model_manifest.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D'],inplace=True,errors='ignore') 
    model_dir = ps.get_directory(version) 
    model_manifest.to_csv(model_dir+'_training_summary_table.csv',index=False)
    model_manifest.to_csv(OUTPUT_DIR+'_training_summary_table.csv',index=False)

def get_mouse_summary_table(version):
    model_dir = ps.get_directory(version)
    return pd.read_csv(model_dir+'_mouse_summary_table.csv').set_index('donor_id')

def build_mouse_summary_table(version):
    ophys = ps.build_model_manifest(version)
    mouse = ophys.groupby('donor_id').mean()
    mouse['cre_line'] = [ophys.query('donor_id ==@donor').iloc[0]['cre_line'] for donor in mouse.index.values]
    midpoint = np.mean(ophys['strategy_dropout_index'])
    mouse['strategy'] = ['visual' if x > midpoint else 'timing' for x in mouse.strategy_dropout_index]
    mouse.drop(columns = [
        'ophys_session_id',
        'behavior_session_id',
        'container_workflow_state',
        'session_type',
        'date_of_acquisition',
        'isi_experiment_id',
        'age_in_days',
        'published_at',
        'session_tags',
        'failure_tags',
        'prior_exposures_to_session_type',
        'prior_exposures_to_image_set',
        'prior_exposures_to_omissions',
        'session_number',
        'active',
        'passive',
        'behavior_fit_available',
        'container_in_order',
        'full_active_container',
        'visual_strategy_session'
        ], inplace=True, errors='ignore')

    model_dir = ps.get_directory(version) 
    mouse.to_csv(model_dir+ '_mouse_summary_table.csv')
    mouse.to_csv(OUTPUT_DIR+'_mouse_summary_table.csv')
   
def build_all_session_outputs(version, TRAIN=False,verbose=False):
    '''
        Iterates a list of session ids, and builds the results file. 
        If TRAIN, uses the training interface
    '''
    # Get list of sessions     
    if TRAIN:
        output_table = pd.read_csv(OUTPUT_DIR+'_training_summary_table.csv')
    else:
        output_table = pd.read_csv(OUTPUT_DIR+'_summary_table.csv')
    ids = output_table['behavior_session_id'].values

    # Iterate each session
    num_crashed = 0
    for index, id in enumerate(tqdm(ids)):
        try:
            if not os.path.isfile(OUTPUT_DIR+str(id)+".csv"):
                build_session_output(id, version,TRAIN=TRAIN)
        except Exception as e:
            num_crashed +=1
            if verbose:
                print('Session CRASHED: ' + str(id)+' '+output_table.loc[index].session_type)
                print(e)
    print(str(num_crashed) + ' sessions crashed')
    print(str(len(ids) - num_crashed) + ' sessions saved')

def build_list_of_missing_session_outputs(version, TRAIN=False):
    '''
        Iterates a list of session ids, and builds the results file. 
        If TRAIN, uses the training interface
    '''
    # Get list of sessions     
    if TRAIN:
        output_table = pd.read_csv(OUTPUT_DIR+'_training_summary_table.csv')
        fname = '/home/alex.piet/codebase/behavior/licking_behavior/scripts/psy_ids_v'+str(version)+'_missing_output_training.txt'
    else:
        output_table = pd.read_csv(OUTPUT_DIR+'_summary_table.csv')
        fname = '/home/alex.piet/codebase/behavior/licking_behavior/scripts/psy_ids_v'+str(version)+'_missing_output.txt'
    ids = output_table['behavior_session_id'].values

    # Iterate each session
    bad_ids = []
    for index, id in enumerate(tqdm(ids)):
        if not os.path.isfile(OUTPUT_DIR+str(id)+".csv"):
            bad_ids.append(id)
    print(str(len(bad_ids)) + ' sessions with no outputs')

    
    # Filter and save
    np.savetxt(fname, bad_ids)
    return bad_ids 
    
def build_session_output(id,version,TRAIN=False):
    '''
        Saves an analysis file in <output_dir> for the model fit of session <id> 
        Extends model weights to be constant during licking bouts
    '''
    # Get Stimulus Info, append model free metrics
    session = pgt.get_data(id)
    pm.get_metrics(session)

    # Load Model fit
    fit = ps.load_fit(id, version=version)
 
    # include when licking bout happened
    session.stimulus_presentations['in_bout'] = fit['psydata']['full_df']['in_bout']
 
    # include model weights
    weights = ps.get_weights_list(fit['weights'])
    for wdex, weight in enumerate(weights):
        session.stimulus_presentations.at[~session.stimulus_presentations.in_bout.values.astype(bool), weight] = fit['wMode'][wdex,:]

    # Iterate value from start of bout forward
    session.stimulus_presentations.fillna(method='ffill', inplace=True)

    # Clean up Stimulus Presentations
    model_output = session.stimulus_presentations.copy()
    model_output.drop(columns=['duration', 'end_frame', 'image_set','index', 
        'orientation', 'start_frame', 'start_time', 'stop_time', 'licks', 
        'rewards', 'time_from_last_lick', 'time_from_last_reward', 
        'time_from_last_change', 'mean_running_speed', 'num_bout_start', 
        'num_bout_end','change_with_lick','change_without_lick',
        'non_change_with_lick','non_change_without_lick'
        ],inplace=True,errors='ignore') 

    # Add binary engagement
    model_output['engaged'] = [(x=='high-lick,low-reward') or (x=='high-lick,high-reward') for x in model_output['flash_metrics_labels']]

    # Clean up some names
    model_output = model_output.rename(columns={
        'in_bout':'in_lick_bout',
        'bout_end':'lick_bout_end',
        'bout_start':'lick_bout_start',
        'bout_rate':'lick_bout_rate',
        'hit_bout':'rewarded_lick_bout',
        'high_lick':'high_lick_state',
        'high_reward':'high_reward_state'
        })

    # Save out dataframe
    model_output.to_csv(OUTPUT_DIR+str(id)+'.csv') 

def build_list_of_model_crashes(version=None):
    '''
        Builds and returns a dataframe that contains information on whether a model fit is available for each 
        behavior_session_id in the manifest. 
        version, version of model to load. If none is given, loads whatever is saved in OUTPUT_DIR
    '''
    manifest = pgt.get_ophys_manifest().query('active').copy()
    if version is None:
        directory = OUTPUT_DIR
    else:
        directory = ps.get_directory(version)
    model_manifest = pd.read_csv(directory+'_summary_table.csv')
    crash=manifest[~manifest.behavior_session_id.isin(model_manifest.behavior_session_id)]  
    return crash

def build_list_of_train_model_crashes(version=None):
    '''
        Builds and returns a dataframe that contains information on whether a model fit is available for each 
        behavior_session_id in the training_manifest. 
        version, version of model to load. If none is given, loads whatever is saved in OUTPUT_DIR
    '''
    manifest = pgt.get_training_manifest().query('active').copy()
    if version is None:
        directory = OUTPUT_DIR
    else:
        directory = ps.get_directory(version)
    model_manifest = pd.read_csv(directory+'_training_summary_table.csv')
    crash=manifest[~manifest.behavior_session_id.isin(model_manifest.behavior_session_id)]  
    return crash





