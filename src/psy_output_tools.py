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

def get_model_inventory(version,include_4x2=False):
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
    manifest = pgt.get_ophys_manifest(include_4x2=include_4x2).copy()

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

def build_inventory_table(vrange=[20,22],include_4x2=False):
    '''
        Returns a dataframe with the number of sessions fit and missing for each model version
    '''
    # Get list of versions
    versions = get_model_versions(vrange)
    
    # Get inventory for each version
    inventories = []
    for v in versions:
        inventories.append(get_model_inventory(v,include_4x2=include_4x2))

    # Combine inventories into dataframe
    table = pd.DataFrame(inventories)
    table = table.drop(columns=['fit_sessions','missing_sessions','with_strategy_df','without_strategy_df']).set_index('version')
    return table 

def make_version(VERSION):
    '''
        Makes directories and saves model parameters 
    '''
 
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
        Saves out the model summary table as a csv file 
    '''
    print('Building Summary Table')
    print('Loading Model Fits')
    summary_df = build_core_table(version)

    print('Loading image by image information')
    summary_df = add_time_aligned_session_info(summary_df,version)
    return summary_df  

    print('Adding engagement information') 
    summary_df = add_engagement_metrics(summary_df) # TODO Issue #202

    print('Creating strategy matched subset')
    summary_df = build_strategy_matched_subset(summary_df)# TODO #203

    # Analyze by order of sessions
    summary_df = add_container_processing(summary_df) # TODO #204

    print('Saving')
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    summary_df.to_pickle(model_dir+'_summary_table.pkl')


def build_core_table(version,container_in_order=False, full_active_container=False,include_4x2=False):
    '''
        Builds a summary_df of model results, each row is a behavioral session. 

        version (int), behavioral model version        
        container_in_order (bool), then only returns sessions that come from a 
            container that was collected in order. The container does not 
            need to be complete, as long as the sessions that are present were collected in order
        full_active_container (bool), then only returns sessions that come from a 
            container with 4 active sessions.
        include_4x2 (bool), whether to include the 4 areas 2 depths dataset. 
    
    '''
    summary_df = pgt.get_ophys_manifest(include_4x2=include_4x2).copy()

    summary_df['behavior_fit_available'] = summary_df['trained_A'] #Just copying the column size
    for index, row in tqdm(summary_df.iterrows(),total=summary_df.shape[0]):
        try:
            fit = ps.load_fit(row.behavior_session_id,version=version)
        except:
            summary_df.at[index,'behavior_fit_available'] = False
        else:
            summary_df.at[index, 'behavior_fit_available'] = True
            summary_df.at[index, 'session_roc'] = ps.compute_model_roc(fit)

            # Get Strategy indices
            model_dex, taskdex,timingdex = ps.get_timing_index_fit(fit,return_all=True) #TODO, Issue #173
            summary_df.at[index,'strategy_dropout_index'] = model_dex
            summary_df.at[index,'visual_only_dropout_index'] = taskdex
            summary_df.at[index,'timing_only_dropout_index'] = timingdex

            # For each strategy add the hyperparameter, dropout score, and average weight
            dropout_dict = ps.get_session_dropout(fit) #TODO, Issue #173
            sigma = fit['hyp']['sigma']
            wMode = fit['wMode']
            weights = ps.get_weights_list(fit['weights'])
            for dex, weight in enumerate(weights):
                summary_df.at[index, 'prior_'+weight] =sigma[dex]
            for dex, weight in enumerate(weights):
                summary_df.at[index, 'dropout_'+weight] = dropout_dict[weight]
            for dex, weight in enumerate(weights):
                summary_df.at[index, 'avg_weight_'+weight] = np.mean(wMode[dex,:])

    # Return only for sessions with fits
    print(str(len(summary_df.query('not behavior_fit_available')))+" sessions without model fits")
    summary_df = summary_df.query('behavior_fit_available').copy()
    
    # Compute weight based index, classify session
    summary_df['strategy_weight_index']   = summary_df['avg_weight_task0'] - summary_df['avg_weight_timing1D'] # TODO Issue #201
    summary_df['visual_strategy_session'] = -summary_df['visual_only_dropout_index'] > -summary_df['timing_only_dropout_index']

    return summary_df

def add_container_processing(summary_df):
    return summary_df
    # Annotate containers
    # TODO Issue, #204
    in_order = []
    four_active = []
    for index, mouse in enumerate(np.array(summary_df['ophys_container_id'].unique())):
        this_df = summary_df.query('ophys_container_id == @mouse')
        stages = this_df.session_number.values
        if np.all(stages ==sorted(stages)):
            in_order.append(mouse)
        if len(this_df) == 4:
            four_active.append(mouse)
    summary_df['container_in_order'] = summary_df.apply(lambda x: x['ophys_container_id'] in in_order, axis=1)
    summary_df['full_active_container'] = summary_df.apply(lambda x: x['ophys_container_id'] in four_active,axis=1)

    # Filter and report outcomes
    if container_in_order:
        n_remove = len(summary_df.query('not container_in_order'))
        print(str(n_remove) + " sessions out of order")
        summary_df = summary_df.query('container_in_order')
    if full_active_container:
        n_remove = len(summary_df.query('not full_active_container'))
        print(str(n_remove) + " sessions from incomplete active containers")
        summary_df = summary_df.query('full_active_container')
        if not (np.mod(len(summary_df),4) == 0):
            raise Exception('Filtered for full containers, but dont seem to have the right number')
    n = len(summary_df)
    print(str(n) + " sessions returned")
    
    return summary_df


# TODO, Clean up, Issue #202
def engagement_for_summary_table(fit, lick_threshold=0.1, reward_threshold=1/90, use_bouts=True,win_dur=320, win_type='triang'):
    fit['psydata']['full_df']['bout_rate'] = fit['psydata']['full_df']['bout_start'].rolling(win_dur,min_periods=1, win_type=win_type).mean()/.75
    #fit['psydata']['full_df']['high_lick'] = [True if x > lick_threshold else False for x in fit['psydata']['full_df']['bout_rate']] 
    fit['psydata']['full_df']['reward_rate'] = fit['psydata']['full_df']['hits'].rolling(win_dur,min_periods=1,win_type=win_type).mean()/.75
    #fit['psydata']['full_df']['high_reward'] = [True if x > reward_threshold else False for x in fit['psydata']['full_df']['reward_rate']] 
    #fit['psydata']['full_df']['flash_metrics_epochs'] = [0 if (not x[0]) & (not x[1]) else 1 if x[1] else 2 for x in zip(fit['psydata']['full_df']['high_lick'], fit['psydata']['full_df']['high_reward'])]
    #fit['psydata']['full_df']['flash_metrics_labels'] = ['low-lick,low-reward' if x==0  else 'high-lick,high-reward' if x==1 else 'high-lick,low-reward' for x in fit['psydata']['full_df']['flash_metrics_epochs']]
    #fit['psydata']['full_df']['engaged'] = [(x=='high-lick,low-reward') or (x=='high-lick,high-reward') for x in fit['psydata']['full_df']['flash_metrics_labels']]
    fit['psydata']['full_df']['engaged'] = [x > reward_threshold for x in fit['psydata']['full_df']['reward_rate']]
    return fit


def add_engagement_metrics(summary_df):
    # TODO, Issues #202, engaged gets added later, so I should probably add this to add_engagement_metrics
    #fit = engagement_for_summary_table(fit) # Should I combine this with add_engagement_metrics?
    # summary_df.at[index, 'fraction_engaged'] = fit['psydata']['full_df']['engaged'].mean() # should I combine this with add_engagement_metrics
    
    # TODO, Issues #202,make these all engaged/disengaged couplets, or all engaged, then all disengaged
    # Add Engaged specific metrics
    summary_df['visual_weight_index_engaged'] = [np.mean(summary_df.loc[x]['weight_task0'][summary_df.loc[x]['engaged'] == True]) for x in summary_df.index.values] 
    summary_df['timing_weight_index_engaged'] = [np.mean(summary_df.loc[x]['weight_timing1D'][summary_df.loc[x]['engaged'] == True]) for x in summary_df.index.values]
    summary_df['omissions_weight_index_engaged'] = [np.mean(summary_df.loc[x]['weight_omissions'][summary_df.loc[x]['engaged'] == True]) for x in summary_df.index.values]
    summary_df['omissions1_weight_index_engaged'] =[np.mean(summary_df.loc[x]['weight_omissions1'][summary_df.loc[x]['engaged'] == True]) for x in summary_df.index.values]
    summary_df['bias_weight_index_engaged'] = [np.mean(summary_df.loc[x]['weight_bias'][summary_df.loc[x]['engaged'] == True]) for x in summary_df.index.values]
    summary_df['visual_weight_index_disengaged'] = [np.mean(summary_df.loc[x]['weight_task0'][summary_df.loc[x]['engaged'] == False]) for x in summary_df.index.values] 
    summary_df['timing_weight_index_disengaged'] = [np.mean(summary_df.loc[x]['weight_timing1D'][summary_df.loc[x]['engaged'] == False]) for x in summary_df.index.values]
    summary_df['omissions_weight_index_disengaged']=[np.mean(summary_df.loc[x]['weight_omissions'][summary_df.loc[x]['engaged']== False]) for x in summary_df.index.values]
    summary_df['omissions1_weight_index_disengaged']=[np.mean(summary_df.loc[x]['weight_omissions1'][summary_df.loc[x]['engaged']==False]) for x in summary_df.index.values]
    summary_df['bias_weight_index_disengaged'] = [np.mean(summary_df.loc[x]['weight_bias'][summary_df.loc[x]['engaged'] == False]) for x in summary_df.index.values]
    summary_df['strategy_weight_index_engaged'] = summary_df['visual_weight_index_engaged'] - summary_df['timing_weight_index_engaged']
    summary_df['strategy_weight_index_disengaged'] = summary_df['visual_weight_index_disengaged'] - summary_df['timing_weight_index_disengaged']
    columns = {'lick_bout_rate','reward_rate','engaged','lick_hit_fraction_rate','hit','miss','FA','CR'}
    for column in columns:  
        if column is not 'engaged':
            summary_df[column+'_engaged'] = [np.mean(summary_df.loc[x][column][summary_df.loc[x]['engaged'] == True]) for x in summary_df.index.values]
            summary_df[column+'_disengaged'] = [np.mean(summary_df.loc[x][column][summary_df.loc[x]['engaged'] == False]) for x in summary_df.index.values]
    summary_df['RT_engaged'] =    [np.nanmean(summary_df.loc[x]['RT'][summary_df.loc[x]['engaged'] == True]) for x in summary_df.index.values]
    summary_df['RT_disengaged'] = [np.nanmean(summary_df.loc[x]['RT'][summary_df.loc[x]['engaged'] == False]) for x in summary_df.index.values]
    return summary_df

def add_time_aligned_session_info(summary_df,version):
    
    # Initializing empty columns
    weight_columns = {'bias','task0','omissions','omissions1','timing1D'} #TODO Dont hard code
    columns = {'hit','miss','FA','CR','change', 'lick_bout_rate','reward_rate','RT','engaged','lick_bout_start'} 
    for column in weight_columns:
        summary_df['weight_'+column] = [[]]*len(summary_df)
    for column in columns:
        summary_df[column] = [[]]*len(summary_df)      
    summary_df['strategy_weight_index_by_image'] = [[]]*len(summary_df)
    summary_df['lick_hit_fraction_rate'] = [[]]*len(summary_df)

    crash = 0
    for index, row in tqdm(summary_df.iterrows(),total=summary_df.shape[0]):
        try:
            strategy_dir = pgt.get_directory(version, subdirectory='strategy_df')
            session_df = pd.read_csv(strategy_dir+str(row.behavior_session_id)+'.csv')
        except Exception as e:
            crash +=1
            print(e)
            for column in weight_columns:
                summary_df.at[index, 'weight_'+column] = np.array([np.nan]*4800)
            for column in columns:
                summary_df.at[index, column] = np.array([np.nan]*4800) 
            summary_df.at[index, column] = np.array([np.nan]*4800)
        else:
            # Define response times
            session_df['hit']  = session_df['rewarded']
            session_df['miss'] = session_df['change'] & ~session_df['rewarded']
            session_df['FA']   = session_df['lick_bout_start'] & session_df['rewarded']
            session_df['CR']   = ~session_df['lick_bout_start'] & ~session_df['change']

            # Add session level metrics
            summary_df.at[index,'num_hits'] = session_df['hit'].sum()
            summary_df.at[index,'num_miss'] = session_df['miss'].sum()
            summary_df.at[index,'num_fa'] = session_df['FA'].sum()
            summary_df.at[index,'num_cr'] = session_df['CR'].sum()
            #summary_df.at[index,'num_aborts'] = ??? #TODO
            summary_df.at[index,'num_lick_bouts'] = session_df['lick_bout_start'].sum()
            summary_df.at[index,'lick_fraction'] = session_df['lick_bout_start'].mean()
            summary_df.at[index,'lick_hit_fraction'] = session_df['rewarded'].sum()/session_df['lick_bout_start'].sum() 
            summary_df.at[index,'trial_hit_fraction'] = session_df['rewarded'].sum()/session_df['change'].sum() 

            # Add time aligned information
            for column in weight_columns:
                summary_df.at[index, 'weight_'+column] = pgt.get_clean_rate(session_df[column].values)
            for column in columns:
                summary_df.at[index, column] = pgt.get_clean_rate(session_df[column].values)
            summary_df.at[index,'strategy_weight_index_by_image'] = pgt.get_clean_rate(session_df['task0'].values) - pgt.get_clean_rate(session_df['timing1D'].values) 
            summary_df.at[index,'lick_hit_fraction_rate'] = pgt.get_clean_rate(session_df['lick_hit_fraction'].values)

    if crash > 0:
        print(str(crash) + ' sessions crashed')
    return summary_df 


def build_strategy_matched_subset(summary_df):
    # TODO, Issue #203
    print('Warning, strategy matched subset is outdated')
    summary_df['strategy_matched'] = True
    summary_df.loc[(summary_df['cre_line'] == "Slc17a7-IRES2-Cre")&(summary_df['visual_only_dropout_index'] < -10),'strategy_matched'] = False
    summary_df.loc[(summary_df['cre_line'] == "Vip-IRES-Cre")&(summary_df['timing_only_dropout_index'] < -15)&(summary_df['timing_only_dropout_index'] > -20),'strategy_matched'] = False
    return summary_df


def get_mouse_summary_table(version):
    model_dir = pgt.get_directory(version,subdirectory='summary')
    return pd.read_pickle(model_dir+'_mouse_summary_table.pkl').set_index('donor_id')


def build_mouse_summary_table(version):
    ophys = ps.build_summary_table(version)
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
        Saves out the training table as a csv file 
    '''
    raise Exception('Outdated, Issue #92')
    summary_df = build_model_training_table(version)
    summary_df.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D'],inplace=True,errors='ignore') 
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    summary_df.to_pickle(model_dir+'_training_summary_table.pkl')


def build_model_training_table(version=None,verbose=False):
    '''
        Builds a manifest of model results
        Each row is a behavior_session_id
        
        if verbose, logs each crashed session id
        if use_full_ophys, uses the full model for ophys sessions (includes omissions)
    
    '''
    raise Exception('Outdated, Issue #92')
    return manifest


