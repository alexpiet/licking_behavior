import os
import json
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

import psy_tools as ps
import psy_general_tools as pgt
import psy_metrics_tools as pm


def get_model_versions(vrange=[20,22]):
    '''
        Returns a sorted list of behavior model versions
    '''
    
    # Get all versions
    behavior_dir = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/'
    versions = os.listdir(behavior_dir)
    versions = [x for x in versions if x.startswith('psy_fits_v')]
    version_numbers = [int(x[10:]) for x in versions]
    
    # Filter for versions in range
    out_versions = []
    for dex, val in enumerate(np.arange(vrange[0], vrange[1])):
        if val in version_numbers:
            out_versions.append('psy_fits_v'+str(val))

    # Display results
    print('Available Behavior Model versions')
    for v in out_versions:
        print(v)
    print('')
    return out_versions

def get_model_inventory(version):
    '''
        Takes the version as either a number of string 'psy_fits_v<>' and
        returns a dictionary of missing and fit sessions
    '''
   
    # Input handling 
    if isinstance(version, str):
        version_num = version[10:]
    else:
        version_num = version
        version = 'psy_fits_v'+str(version_num)

    # Get information on what SHOULD be available
    manifest = pgt.get_ophys_manifest().copy()

    # Check what is actually available. 
    fit_directory=pgt.get_directory(version_num,subdirectory='fits')
    df_directory=pgt.get_directory(version_num,subdirectory='strategy_df') 
    for index, row in manifest.iterrows():
        fit_filename = fit_directory + str(row.behavior_session_id) + ".pkl"         
        manifest.at[index, 'behavior_fit_available'] = os.path.exists(fit_filename)

        summary_filename = df_directory+ str(row.behavior_session_id)+'.csv'
        manifest.at[index, 'strategy_df_available'] = os.path.exists(summary_filename)

    # Summarize inventory for this model version
    inventory = {}    
    inventory['fit_sessions'] = manifest.query('behavior_fit_available == True')['behavior_session_id']
    inventory['missing_sessions'] = manifest.query('behavior_fit_available != True')['behavior_session_id']
    inventory['with_strategy_df'] = manifest.query('strategy_df_available == True')['behavior_session_id']
    inventory['without_strategy_df'] = manifest.query('strategy_df_available != True')['behavior_session_id']
    inventory['num_sessions'] = len(manifest)
    inventory['num_fit'] = len(inventory['fit_sessions'])
    inventory['num_missing'] = len(inventory['missing_sessions'])
    inventory['num_with_strategy_df'] = len(inventory['with_strategy_df'])
    inventory['num_without_strategy_df'] = len(inventory['without_strategy_df'])
    inventory['version'] = version
    return inventory

def build_inventory_table(vrange=[20,22]):
    '''
        Returns a dataframe with the number of sessions fit and missing for each model version
    '''
    # Get list of versions
    versions = get_model_versions(vrange)
    
    # Get inventory for each version
    inventories = []
    for v in versions:
        inventories.append(get_model_inventory(v))

    # Combine inventories into dataframe
    table = pd.DataFrame(inventories)
    table = table.drop(columns=['fit_sessions','missing_sessions','with_strategy_df','without_strategy_df']).set_index('version')
    return table 

def make_version(VERSION):
    '''
        Saves out two text files with lists of all behavior_session_ids for ophys and training sessions in the manifest
        Only includes active sessions
    '''
    # Get manifest
    manifest = pgt.get_ophys_manifest()
    #training = pgt.get_training_manifest()
 
    # Make appropriate folders
    print('Making directory structure') 
    root_directory  = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/'
    directory = root_directory+'psy_fits_v'+str(VERSION)
    if not os.path.isdir(directory):
        os.mkdir(directory)
        os.mkdir(directory+'/figures_summary')
        os.mkdir(directory+'/figures_sessions')
        os.mkdir(directory+'/figures_training')
        os.mkdir(directory+'/session_fits')
        os.mkdir(directory+'/session_clusters')
        os.mkdir(directory+'/session_strategy_df')
        os.mkdir(directory+'/summary_data')
        os.mkdir(directory+'/psytrack_logs')
    else:
        print('directory already exists')
    
    # Save Version Parameters
    print('Saving parameter json')

    save_version_parameters(VERSION)
    print('Done!')

def save_version_parameters(VERSION):
    git_branch = subprocess.check_output(['git','branch','--show-current'], cwd='/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/src').strip().decode() 
    git_hash = subprocess.check_output(['git','rev-parse','--short','HEAD'], cwd='/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/src').strip().decode()
    format_options = {
                'fit_bouts':True,
                'timing0/1':True,
                'mean_center':True,
                'timing_params':[-5,4],
                'timing_params_session':[-5,4],
                'ignore_trial_errors':False,
                'num_cv_folds':10,
                'git_commit_hash': git_hash,
                'git_branch':git_branch
                }
    
    json_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v'+str(VERSION)+'/summary_data/behavior_model_params.json'
    with open(json_path, 'w') as json_file:
        json.dump(format_options, json_file, indent=4)

def get_ophys_summary_table(version):
    model_dir = pgt.get_directory(version,subdirectory='summary')
    return pd.read_pickle(model_dir+'_summary_table.pkl')

def build_summary_table(version):
    ''' 
        Saves out the model manifest as a csv file 
    '''
    print('Building Summary Table')
    print('Loading Model Fits')
    manifest = ps.build_model_manifest(version=version,container_in_order=False)

    #this are in time units of bouts, we need time-aligned weights
    manifest.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D','weight_omissions'],inplace=True) 
    print('Loading behavioral information')
    manifest = add_time_aligned_session_info(manifest,version)
    manifest = build_strategy_matched_subset(manifest)
    manifest = add_engagement_metrics(manifest)

    print('Saving')
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    manifest.to_pickle(model_dir+'_summary_table.pkl')


def build_model_manifest(version=None,container_in_order=False, full_active_container=False,verbose=False):
    '''
        Builds a manifest of model results
        Each row is a Behavior_session_id
        
        if container_in_order, then only returns sessions that come from a container that was collected in order. The container
            does not need to be complete, as long as the sessions that are present were collected in order
        if full_active_container, then only returns sessions that come from a container with 4 active sessions. 
        if verbose, logs each crashed session id
    
    '''
    manifest = pgt.get_ophys_manifest().copy()
    #directory=pgt.get_directory(version,subdirectory='fits') 

    manifest['behavior_fit_available'] = manifest['trained_A'] #Just copying the column size
    first = True
    crashed = 0
    for index, row in tqdm(manifest.iterrows(),total=manifest.shape[0]):
        try:
            fit = ps.load_fit(row.behavior_session_id,version=version)
        except:
            if verbose:
                print(str(row.behavior_session_id)+" crash")
            manifest.at[index,'behavior_fit_available'] = False
            crashed +=1
        else:
            fit = engagement_for_model_manifest(fit) 
            manifest.at[index,'behavior_fit_available'] = True
            manifest.at[index, 'num_hits'] = np.sum(fit['psydata']['hits'])
            manifest.at[index, 'num_fa'] = np.sum(fit['psydata']['false_alarms'])
            manifest.at[index, 'num_cr'] = np.sum(fit['psydata']['correct_reject'])
            manifest.at[index, 'num_miss'] = np.sum(fit['psydata']['misses'])
            manifest.at[index, 'num_aborts'] = np.sum(fit['psydata']['aborts'])
            manifest.at[index, 'fraction_engaged'] = fit['psydata']['full_df']['engaged'].mean() 
            sigma = fit['hyp']['sigma']
            wMode = fit['wMode']
            weights = get_weights_list(fit['weights'])
            manifest.at[index,'session_roc'] = ps.compute_model_roc(fit)
            manifest.at[index,'lick_fraction'] = ps.get_lick_fraction(fit)
            #manifest.at[index,'lick_fraction_1st_half'] = get_lick_fraction(fit,first_half=True)
            #manifest.at[index,'lick_fraction_2nd_half'] = get_lick_fraction(fit,second_half=True)
            manifest.at[index,'lick_hit_fraction'] = ps.get_hit_fraction(fit)
            #manifest.at[index,'lick_hit_fraction_1st_half'] = get_hit_fraction(fit,first_half=True)
            #manifest.at[index,'lick_hit_fraction_2nd_half'] = get_hit_fraction(fit,second_half=True)
            manifest.at[index,'trial_hit_fraction'] = ps.get_trial_hit_fraction(fit)
            #manifest.at[index,'trial_hit_fraction_1st_half'] = get_trial_hit_fraction(fit,first_half=True)
            #manifest.at[index,'trial_hit_fraction_2nd_half'] = get_trial_hit_fraction(fit,second_half=True)
   
            model_dex, taskdex,timingdex = ps.get_timing_index_fit(fit,return_all=True)
            manifest.at[index,'strategy_dropout_index'] = model_dex
            manifest.at[index,'visual_only_dropout_index'] = taskdex
            manifest.at[index,'timing_only_dropout_index'] = timingdex

            dropout_dict = ps.get_session_dropout(fit)
            for dex, weight in enumerate(weights):
                manifest.at[index, 'prior_'+weight] =sigma[dex]
                manifest.at[index, 'dropout_'+weight] = dropout_dict[weight]
                manifest.at[index, 'avg_weight_'+weight] = np.mean(wMode[dex,:])
                #manifest.at[index, 'avg_weight_'+weight+'_1st_half'] = np.mean(wMode[dex,fit['psydata']['flash_ids']<2400])
                #manifest.at[index, 'avg_weight_'+weight+'_2nd_half'] = np.mean(wMode[dex,fit['psydata']['flash_ids']>=2400])
                if first: 
                    manifest['weight_'+weight] = [[]]*len(manifest)
                manifest.at[index, 'weight_'+str(weight)] = wMode[dex,:]  
            first = False
    print(str(crashed)+ " sessions without model fits")

    manifest = manifest.query('behavior_fit_available').copy()
    manifest['strategy_weight_index']           = manifest['avg_weight_task0'] - manifest['avg_weight_timing1D']
    #manifest['strategy_weight_index_1st_half']  = manifest['avg_weight_task0_1st_half'] - manifest['avg_weight_timing1D_1st_half']
    #manifest['strategy_weight_index_2nd_half']  = manifest['avg_weight_task0_2nd_half'] - manifest['avg_weight_timing1D_2nd_half']
    manifest['visual_strategy_session']         = -manifest['visual_only_dropout_index'] > -manifest['timing_only_dropout_index']

    # Annotate containers
    return manifest # TODO Issue, #149
    in_order = []
    four_active = []
    for index, mouse in enumerate(np.array(manifest['ophys_container_id'].unique())):
        this_df = manifest.query('ophys_container_id == @mouse')
        stages = this_df.session_number.values
        if np.all(stages ==sorted(stages)):
            in_order.append(mouse)
        if len(this_df) == 4:
            four_active.append(mouse)
    manifest['container_in_order'] = manifest.apply(lambda x: x['ophys_container_id'] in in_order, axis=1)
    manifest['full_active_container'] = manifest.apply(lambda x: x['ophys_container_id'] in four_active,axis=1)

    # Filter and report outcomes
    if container_in_order:
        n_remove = len(manifest.query('not container_in_order'))
        print(str(n_remove) + " sessions out of order")
        manifest = manifest.query('container_in_order')
    if full_active_container:
        n_remove = len(manifest.query('not full_active_container'))
        print(str(n_remove) + " sessions from incomplete active containers")
        manifest = manifest.query('full_active_container')
        if not (np.mod(len(manifest),4) == 0):
            raise Exception('Filtered for full containers, but dont seem to have the right number')
    n = len(manifest)
    print(str(n) + " sessions returned")
    
    return manifest


# TODO, Clean up, Issue #149
def engagement_for_model_manifest(fit, lick_threshold=0.1, reward_threshold=1/90, use_bouts=True,win_dur=320, win_type='triang'):
    fit['psydata']['full_df']['bout_rate'] = fit['psydata']['full_df']['bout_start'].rolling(win_dur,min_periods=1, win_type=win_type).mean()/.75
    #fit['psydata']['full_df']['high_lick'] = [True if x > lick_threshold else False for x in fit['psydata']['full_df']['bout_rate']] 
    fit['psydata']['full_df']['reward_rate'] = fit['psydata']['full_df']['hits'].rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75
    #fit['psydata']['full_df']['high_reward'] = [True if x > reward_threshold else False for x in fit['psydata']['full_df']['reward_rate']] 
    #fit['psydata']['full_df']['flash_metrics_epochs'] = [0 if (not x[0]) & (not x[1]) else 1 if x[1] else 2 for x in zip(fit['psydata']['full_df']['high_lick'], fit['psydata']['full_df']['high_reward'])]
    #fit['psydata']['full_df']['flash_metrics_labels'] = ['low-lick,low-reward' if x==0  else 'high-lick,high-reward' if x==1 else 'high-lick,low-reward' for x in fit['psydata']['full_df']['flash_metrics_epochs']]
    #fit['psydata']['full_df']['engaged'] = [(x=='high-lick,low-reward') or (x=='high-lick,high-reward') for x in fit['psydata']['full_df']['flash_metrics_labels']]
    fit['psydata']['full_df']['engaged'] = [x > reward_threshold for x in fit['psydata']['full_df']['reward_rate']]
    return fit


def add_engagement_metrics(manifest):
    # Add Engaged specific metrics
    manifest['visual_weight_index_engaged'] = [np.mean(manifest.loc[x]['weight_task0'][manifest.loc[x]['engaged'] == True]) for x in manifest.index.values] 
    manifest['timing_weight_index_engaged'] = [np.mean(manifest.loc[x]['weight_timing1D'][manifest.loc[x]['engaged'] == True]) for x in manifest.index.values]
    manifest['omissions_weight_index_engaged'] = [np.mean(manifest.loc[x]['weight_omissions'][manifest.loc[x]['engaged'] == True]) for x in manifest.index.values]
    manifest['omissions1_weight_index_engaged'] =[np.mean(manifest.loc[x]['weight_omissions1'][manifest.loc[x]['engaged'] == True]) for x in manifest.index.values]
    manifest['bias_weight_index_engaged'] = [np.mean(manifest.loc[x]['weight_bias'][manifest.loc[x]['engaged'] == True]) for x in manifest.index.values]
    manifest['visual_weight_index_disengaged'] = [np.mean(manifest.loc[x]['weight_task0'][manifest.loc[x]['engaged'] == False]) for x in manifest.index.values] 
    manifest['timing_weight_index_disengaged'] = [np.mean(manifest.loc[x]['weight_timing1D'][manifest.loc[x]['engaged'] == False]) for x in manifest.index.values]
    manifest['omissions_weight_index_disengaged']=[np.mean(manifest.loc[x]['weight_omissions'][manifest.loc[x]['engaged']== False]) for x in manifest.index.values]
    manifest['omissions1_weight_index_disengaged']=[np.mean(manifest.loc[x]['weight_omissions1'][manifest.loc[x]['engaged']==False]) for x in manifest.index.values]
    manifest['bias_weight_index_disengaged'] = [np.mean(manifest.loc[x]['weight_bias'][manifest.loc[x]['engaged'] == False]) for x in manifest.index.values]
    manifest['strategy_weight_index_engaged'] = manifest['visual_weight_index_engaged'] - manifest['timing_weight_index_engaged']
    manifest['strategy_weight_index_disengaged'] = manifest['visual_weight_index_disengaged'] - manifest['timing_weight_index_disengaged']
    columns = {'lick_bout_rate','reward_rate','engaged','lick_hit_fraction_rate','hit','miss','FA','CR'}
    for column in columns:  
        if column is not 'engaged':
            manifest[column+'_engaged'] = [np.mean(manifest.loc[x][column][manifest.loc[x]['engaged'] == True]) for x in manifest.index.values]
            manifest[column+'_disengaged'] = [np.mean(manifest.loc[x][column][manifest.loc[x]['engaged'] == False]) for x in manifest.index.values]
    manifest['RT_engaged'] =    [np.nanmean(manifest.loc[x]['RT'][manifest.loc[x]['engaged'] == True]) for x in manifest.index.values]
    manifest['RT_disengaged'] = [np.nanmean(manifest.loc[x]['RT'][manifest.loc[x]['engaged'] == False]) for x in manifest.index.values]
    return manifest

def add_time_aligned_session_info(manifest,version):
    weight_columns = {'bias','task0','omissions','omissions1','timing1D'}
    for column in weight_columns:
        manifest['weight_'+column] = [[]]*len(manifest)
    columns = {'hit','miss','FA','CR','change', 'lick_bout_rate','reward_rate','RT','engaged','lick_bout_start'} 
    for column in columns:
        manifest[column] = [[]]*len(manifest)      
    manifest['lick_hit_fraction_rate'] = [[]]*len(manifest)

    crash = 0
    for index, row in tqdm(manifest.iterrows(),total=manifest.shape[0]):
        try:
            strategy_dir = pgt.get_directory(version, subdirectory='strategy_df')
            session_df = pd.read_csv(strategy_dir+str(row.behavior_session_id)+'.csv')
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
            manifest.at[index,'lick_hit_fraction_rate'] = pgt.get_clean_rate(session_df['lick_hit_fraction'].values)
        except Exception as e:
            crash +=1
            print(e)
            for column in weight_columns:
                manifest.at[index, 'weight_'+column] = np.array([np.nan]*4800)
            for column in columns:
                manifest.at[index, column] = np.array([np.nan]*4800) 
            manifest.at[index, column] = np.array([np.nan]*4800)
    if crash > 0:
        print(str(crash) + ' sessions crashed, consider running build_all_session_outputs')
    return manifest 

def build_strategy_matched_subset(manifest):
    print('Warning, strategy matched subset is outdated')
    manifest['strategy_matched'] = True
    manifest.loc[(manifest['cre_line'] == "Slc17a7-IRES2-Cre")&(manifest['visual_only_dropout_index'] < -10),'strategy_matched'] = False
    manifest.loc[(manifest['cre_line'] == "Vip-IRES-Cre")&(manifest['timing_only_dropout_index'] < -15)&(manifest['timing_only_dropout_index'] > -20),'strategy_matched'] = False
    return manifest


def get_mouse_summary_table(version):
    model_dir = pgt.get_directory(version,subdirectory='summary')
    return pd.read_pickle(model_dir+'_mouse_summary_table.pkl').set_index('donor_id')


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

    model_dir = pgt.get_directory(version,subdirectory='summary') 
    mouse.to_pickle(model_dir+ '_mouse_summary_table.pkl')
  

def get_training_summary_table(version):
    raise Exception('Outdated, Issue #92')
    model_dir = pgt.get_directory(version,subdirectory='summary')
    return pd.read_pickle(model_dir+'_training_summary_table.pkl')


def build_training_summary_table(version):
    ''' 
        Saves out the model manifest as a csv file 
    '''
    raise Exception('Outdated, Issue #92')
    model_manifest = build_model_training_manifest(version)
    model_manifest.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D'],inplace=True,errors='ignore') 
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    model_manifest.to_pickle(model_dir+'_training_summary_table.pkl')


def build_model_training_manifest(version=None,verbose=False):
    '''
        Builds a manifest of model results
        Each row is a behavior_session_id
        
        if verbose, logs each crashed session id
        if use_full_ophys, uses the full model for ophys sessions (includes omissions)
    
    '''
    raise Exception('Outdated, Issue #92')
    manifest = pgt.get_training_manifest().copy()
    directory = pgt.get_directory(version)

    manifest['behavior_fit_available'] = manifest['active'] #Just copying the column size
    first = True
    crashed = 0
    for index, row in tqdm(manifest.iterrows(),total=manifest.shape[0]):
        try:
            fit = load_fit(row.behavior_session_id,version=version)
        except:
            if verbose:
                print(str(row.behavior_session_id)+" crash")
            manifest.at[index,'behavior_fit_available'] = False
            crashed +=1
        else:
            fit = engagement_for_model_manifest(fit) 
            manifest.at[index,'behavior_fit_available'] = True
            manifest.at[index, 'num_hits']  = np.sum(fit['psydata']['hits'])
            manifest.at[index, 'num_fa']    = np.sum(fit['psydata']['false_alarms'])
            manifest.at[index, 'num_cr']    = np.sum(fit['psydata']['correct_reject'])
            manifest.at[index, 'num_miss']  = np.sum(fit['psydata']['misses'])
            manifest.at[index, 'num_aborts']= np.sum(fit['psydata']['aborts'])
            manifest.at[index, 'fraction_engaged'] = fit['psydata']['full_df']['engaged'].mean() 
            sigma = fit['hyp']['sigma']
            wMode = fit['wMode']
            weights = get_weights_list(fit['weights'])
            manifest.at[index,'session_roc'] = compute_model_roc(fit)
            manifest.at[index,'lick_fraction']          = get_lick_fraction(fit)
            manifest.at[index,'lick_fraction_1st_half'] = get_lick_fraction(fit,first_half=True)
            manifest.at[index,'lick_fraction_2nd_half'] = get_lick_fraction(fit,second_half=True)
            manifest.at[index,'lick_hit_fraction']          = get_hit_fraction(fit)
            manifest.at[index,'lick_hit_fraction_1st_half'] = get_hit_fraction(fit,first_half=True)
            manifest.at[index,'lick_hit_fraction_2nd_half'] = get_hit_fraction(fit,second_half=True)
            manifest.at[index,'trial_hit_fraction']          = get_trial_hit_fraction(fit)
            manifest.at[index,'trial_hit_fraction_1st_half'] = get_trial_hit_fraction(fit,first_half=True)
            manifest.at[index,'trial_hit_fraction_2nd_half'] = get_trial_hit_fraction(fit,second_half=True)

            model_dex, taskdex,timingdex = get_timing_index_fit(fit,return_all=True)
            manifest.at[index,'strategy_dropout_index'] = model_dex
            manifest.at[index,'visual_only_dropout_index'] = taskdex
            manifest.at[index,'timing_only_dropout_index'] = timingdex

            if first:
                possible_weights = {'bias','task0','timing1D','omissions','omissions1'}
                for weight in possible_weights: 
                    manifest['weight_'+weight] = [[]]*len(manifest)
                first=False 

            for dex, weight in enumerate(weights):
                manifest.at[index, 'prior_'+weight] =sigma[dex]
                manifest.at[index, 'avg_weight_'+weight] = np.mean(wMode[dex,:])
                manifest.at[index, 'avg_weight_'+weight+'_1st_half'] = np.mean(wMode[dex,fit['psydata']['flash_ids']<2400])
                if len(fit['psydata']['flash_ids']) >=2400:
                    manifest.at[index, 'avg_weight_'+weight+'_2nd_half'] = np.mean(wMode[dex,fit['psydata']['flash_ids']>=2400])
                manifest.at[index, 'weight_'+str(weight)] = wMode[dex,:]  

    manifest = manifest.query('behavior_fit_available').copy()
    manifest['strategy_weight_index']           = manifest['avg_weight_task0'] - manifest['avg_weight_timing1D']
    manifest['strategy_weight_index_1st_half']  = manifest['avg_weight_task0_1st_half'] - manifest['avg_weight_timing1D_1st_half']
    manifest['strategy_weight_index_2nd_half']  = manifest['avg_weight_task0_2nd_half'] - manifest['avg_weight_timing1D_2nd_half']
    manifest['visual_strategy_session']         = -manifest['visual_only_dropout_index'] > -manifest['timing_only_dropout_index']

    n = len(manifest)
    print(str(crashed)+ " sessions crashed")
    print(str(n) + " sessions returned")
    
    return manifest


