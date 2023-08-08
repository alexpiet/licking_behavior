import os
import json
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm

import psy_tools as ps
import psy_general_tools as pgt

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
    inventory['fit_sessions'] = \
        manifest.query('behavior_fit_available == True')['behavior_session_id']
    inventory['missing_sessions'] = \
        manifest.query('behavior_fit_available != True')['behavior_session_id']
    inventory['with_strategy_df'] = \
        manifest.query('strategy_df_available == True')['behavior_session_id']
    inventory['without_strategy_df'] = \
        manifest.query('strategy_df_available != True')['behavior_session_id']
    inventory['num_sessions'] = len(manifest)
    inventory['num_fit'] = len(inventory['fit_sessions'])
    inventory['num_missing'] = len(inventory['missing_sessions'])
    inventory['num_with_strategy_df'] = len(inventory['with_strategy_df'])
    inventory['num_without_strategy_df'] = len(inventory['without_strategy_df'])
    inventory['version'] = version
    return inventory

def build_inventory_table(vrange=[20,22],include_4x2=False):
    '''
        Returns a dataframe with the number of sessions fit and missing for 
        each model version
    '''
    # Get list of versions
    versions = get_model_versions(vrange)
    
    # Get inventory for each version
    inventories = []
    for v in versions:
        inventories.append(get_model_inventory(v,include_4x2=include_4x2))

    # Combine inventories into dataframe
    table = pd.DataFrame(inventories)
    table = table.drop(columns=['fit_sessions','missing_sessions',
        'with_strategy_df','without_strategy_df']).set_index('version')
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
        os.mkdir(directory+'/session_strategy_df')
        os.mkdir(directory+'/session_licks_df')
        os.mkdir(directory+'/summary_data')
        os.mkdir(directory+'/psytrack_logs')
    else:
        print('directory already exists')
    
    # Save Version Parameters
    print('Saving parameter json')

    save_version_parameters(VERSION)
    print('Done!')

def save_version_parameters(VERSION):
    git_branch = subprocess.check_output(['git','branch','--show-current'],\
        cwd='/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/src').strip().decode() 
    git_hash = subprocess.check_output(['git','rev-parse','--short','HEAD'],\
        cwd='/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/src').strip().decode()
    format_options = {
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
    summary_df = build_strategy_labels(summary_df)

    print('Creating strategy matched subset')
    summary_df = build_strategy_matched_subset(summary_df)

    print('Loading image by image information')
    summary_df = add_time_aligned_session_info(summary_df,version)

    print('Adding engagement information') 
    summary_df = add_engagement_metrics(summary_df) 
    summary_df = temporary_engagement_updates(summary_df)


    print('Saving')
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    summary_df.to_pickle(model_dir+'_summary_table.pkl')

    return summary_df 

def temporary_engagement_updates(summary_df):
    summary_df['engagement_v1'] = [[]]*len(summary_df)      
    summary_df['engagement_v2'] = [[]]*len(summary_df)      
    summary_df['engagement_v1'] = summary_df['engaged']
    v2 = []
    for index, row in tqdm(summary_df.iterrows(), total=summary_df.shape[0]):
        this = np.array([x[0] and x[1] > 0.1 for x in zip(row.engagement_v1, row.lick_bout_rate)])
        v2.append(this)
    summary_df['engagement_v2'] = v2
    return summary_df

def build_core_table(version,include_4x2=False):
    '''
        Builds a summary_df of model results, each row is a behavioral session. 

        version (int), behavioral model version        
        include_4x2 (bool), whether to include the 4 areas 2 depths dataset. 
    
    '''
    summary_df = pgt.get_ophys_manifest(include_4x2=include_4x2).copy()

    summary_df['behavior_fit_available'] = summary_df['trained_A'] #copying column size
    for index, row in tqdm(summary_df.iterrows(),total=summary_df.shape[0]):
        try:
            fit = ps.load_fit(row.behavior_session_id,version=version)
        except:
            summary_df.at[index,'behavior_fit_available'] = False
        else:
            summary_df.at[index,'behavior_fit_available'] = True 
            summary_df.at[index,'session_roc'] = ps.compute_model_roc(fit)
            summary_df.at[index,'num_trial_false_alarm'] = \
                np.sum(fit['psydata']['full_df']['false_alarm'])
            summary_df.at[index,'num_trial_correct_reject'] = \
                np.sum(fit['psydata']['full_df']['correct_reject'])

            # Get Strategy indices
            model_dex, taskdex,timingdex = ps.get_timing_index_fit(fit) 
            summary_df.at[index,'strategy_dropout_index'] = model_dex
            summary_df.at[index,'visual_only_dropout_index'] = taskdex
            summary_df.at[index,'timing_only_dropout_index'] = timingdex

            # For each strategy add the hyperparameter, dropout score, and average weight
            dropout_dict_cv = ps.get_session_dropout(fit,cross_validation=True)
            dropout_dict_ev = ps.get_session_dropout(fit,cross_validation=False)
            sigma = fit['hyp']['sigma']
            wMode = fit['wMode']
            weights = ps.get_weights_list(fit['weights'])
            for dex, weight in enumerate(weights):
                summary_df.at[index, 'prior_'+weight] =sigma[dex]
            for dex, weight in enumerate(weights):
                summary_df.at[index, 'dropout_cv_'+weight] = dropout_dict_cv[weight]
                summary_df.at[index, 'dropout_'+weight] = dropout_dict_ev[weight]
            for dex, weight in enumerate(weights):
                summary_df.at[index, 'avg_weight_'+weight] = np.mean(wMode[dex,:])

    # Return only for sessions with fits
    print(str(len(summary_df.query('not behavior_fit_available')))+\
        " sessions without model fits")
    summary_df = summary_df.query('behavior_fit_available').copy()
    
    # Compute classify session
    summary_df['visual_strategy_session'] = -summary_df['visual_only_dropout_index'] > \
        -summary_df['timing_only_dropout_index']

    return summary_df


def add_engagement_metrics(summary_df,min_engaged_fraction=.05):  
    '''
        Adds average value of columns for engaged and disengaged periods
    '''

    # Add Engaged specific metrics
    summary_df['fraction_engaged'] = \
        [np.nanmean(summary_df.loc[x]['engaged']) for x in summary_df.index.values]

    # Add average value of strategy weights split by engagement stats
    columns = {
        'task0':'visual',
        'timing1D':'timing',
        'omissions':'omissions',
        'omissions1':'omissions1',
        'bias':'bias'}
    for k in columns.keys():  
        summary_df[columns[k]+'_weight_index_engaged'] = \
        [np.nanmean(summary_df.loc[x]['weight_'+k][summary_df.loc[x]['engaged'] == True]) 
            if summary_df.loc[x]['fraction_engaged'] > min_engaged_fraction else np.nan 
            for x in summary_df.index.values]
        summary_df[columns[k]+'_weight_index_disengaged'] = \
        [np.nanmean(summary_df.loc[x]['weight_'+k][summary_df.loc[x]['engaged'] == False])
            if summary_df.loc[x]['fraction_engaged'] < 1-min_engaged_fraction else np.nan 
            for x in summary_df.index.values]
    summary_df['strategy_weight_index_engaged'] = \
        summary_df['visual_weight_index_engaged'] -\
        summary_df['timing_weight_index_engaged']
    summary_df['strategy_weight_index_disengaged'] = \
        summary_df['visual_weight_index_disengaged'] -\
        summary_df['timing_weight_index_disengaged']

    # Add average value of columns split by engagement state
    columns = {'lick_bout_rate','reward_rate','lick_hit_fraction_rate','hit',
        'miss','image_false_alarm','image_correct_reject','RT'}
    for column in columns: 
        summary_df[column+'_engaged'] = \
        [np.nanmean(summary_df.loc[x][column][summary_df.loc[x]['engaged'] == True]) 
            if summary_df.loc[x]['fraction_engaged'] > min_engaged_fraction else np.nan
            for x in summary_df.index.values]
        summary_df[column+'_disengaged'] = \
        [np.nanmean(summary_df.loc[x][column][summary_df.loc[x]['engaged'] == False]) 
            if (summary_df.loc[x]['fraction_engaged'] < 1-min_engaged_fraction) &
            (not np.all(np.isnan(summary_df.loc[x][column][summary_df.loc[x]['engaged']==False]))) 
            else np.nan for x in summary_df.index.values]
    return summary_df

def add_time_aligned_session_info(summary_df,version):
    
    # Initializing empty columns
    weight_columns = pgt.get_strategy_list(version)
    columns = {'hit','miss','image_false_alarm','image_correct_reject',
        'is_change', 'omitted', 'lick_bout_rate','reward_rate','RT','engaged',
        'lick_bout_start','image_index','reward_latency'} 
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
            if version <=20:
                session_df = session_df_backwards_compatability(session_df)
        except Exception as e:
            crash +=1
            print(e)
            for column in weight_columns:
                summary_df.at[index, 'weight_'+column] = np.array([np.nan]*4800)
            for column in columns:
                summary_df.at[index, column] = np.array([np.nan]*4800) 
            summary_df.at[index, column] = np.array([np.nan]*4800)
        else: 
            # Add session level metrics
            summary_df.at[index,'num_hits'] = session_df['hit'].sum()
            summary_df.at[index,'num_miss'] = session_df['miss'].sum()
            summary_df.at[index,'num_omission_licks'] = \
                np.sum(session_df['omitted'] & session_df['lick_bout_start']) 
            summary_df.at[index,'num_post_omission_licks'] = \
                np.sum(session_df['omitted'].shift(1,fill_value=False) & \
                session_df['lick_bout_start'])
            summary_df.at[index,'num_late_task_licks'] = \
                np.sum(session_df['is_change'].shift(1,fill_value=False) & \
                session_df['lick_bout_start'])
            summary_df.at[index,'num_changes'] = session_df['is_change'].sum()
            summary_df.at[index,'num_omissions'] = session_df['omitted'].sum()
            summary_df.at[index,'num_image_false_alarm'] = \
                session_df['image_false_alarm'].sum()
            summary_df.at[index,'num_image_correct_reject'] = \
                session_df['image_correct_reject'].sum()
            summary_df.at[index,'num_lick_bouts'] = session_df['lick_bout_start'].sum()
            summary_df.at[index,'lick_fraction'] = session_df['lick_bout_start'].mean()
            summary_df.at[index,'omission_lick_fraction'] = \
                summary_df.at[index,'num_omission_licks']/\
                summary_df.at[index,'num_omissions'] 
            summary_df.at[index,'post_omission_lick_fraction'] = \
                summary_df.at[index,'num_post_omission_licks']/\
                summary_df.at[index,'num_omissions'] 
            summary_df.at[index,'lick_hit_fraction'] = \
                session_df['rewarded'].sum()/session_df['lick_bout_start'].sum() 
            summary_df.at[index,'trial_hit_fraction'] = \
                session_df['rewarded'].sum()/session_df['is_change'].sum() 

            # Add time aligned information
            for column in weight_columns:
                summary_df.at[index, 'weight_'+column] = \
                    pgt.get_clean_rate(session_df[column].values)
            for column in columns:
                summary_df.at[index, column] = \
                    pgt.get_clean_rate(session_df[column].values)
            summary_df.at[index,'lick_hit_fraction_rate'] = \
                pgt.get_clean_rate(session_df['lick_hit_fraction'].values)

            # Compute strategy indexes
            summary_df.at[index,'strategy_weight_index_by_image'] = \
                pgt.get_clean_rate(session_df['task0'].values) - \
                pgt.get_clean_rate(session_df['timing1D'].values) 
            summary_df.at[index,'strategy_weight_index'] = \
                np.nanmean(summary_df.at[index,'strategy_weight_index_by_image'])

    if crash > 0:
        print(str(crash) + ' sessions crashed')
    return summary_df 

def session_df_backwards_compatability(session_df):
    '''
        Starting in version 21 these columns are computed in the session_df
        for backwards compatability I can compute them here
    '''
    session_df['hit'] = [np.nan if (not x[0]) else 1 if (x[1]) else 0 
        for x in zip(session_df['is_change'], session_df['rewarded'])]
    session_df['miss'] = [np.nan if (not x[0]) else 0 if (x[1]) else 1 
        for x in zip(session_df['is_change'],session_df['rewarded'])]
    session_df['image_false_alarm'] = [np.nan if (x[0]) else 1 if (x[1]) else 0 
        for x in zip(session_df['is_change'],session_df['lick_bout_start'])]
    session_df['image_correct_reject'] = \
        [np.nan if (x[0] or (x[2] and not x[1])) else 0 if (x[1]) else 1 
        for x in zip(session_df['is_change'],session_df['lick_bout_start'],\
            session_df['licked'])]

    return session_df


def build_strategy_labels(summary_df):

    # basic split
    summary_df['strategy_labels'] = ['visual' if x else 'timing' \
        for x in summary_df['visual_strategy_session']]

    # no strategy group
    summary_df['strategy_labels_with_none'] = ['visual' if x else 'timing' for x in 
        summary_df['visual_strategy_session']] 
    none_strategies = (summary_df['dropout_task0'] > -7.5)&\
        (summary_df['dropout_timing1D'] > -7.5)
    summary_df.at[none_strategies,'strategy_labels_with_none'] = 'no strategy'

    # mixed strategy group
    quantiles = np.quantile(summary_df['strategy_dropout_index'],[.33,.66])
    if quantiles[1] < 0:
        quantiles[1] = 0
    summary_df['strategy_labels_with_mixed'] = ['visual' if x else 'timing' for x in 
        summary_df['visual_strategy_session']] 
    mixed_strategies = (summary_df['strategy_dropout_index'] > quantiles[0])&\
        (summary_df['strategy_dropout_index'] < quantiles[1])
    summary_df.at[mixed_strategies,'strategy_labels_with_mixed'] = 'mixed'
    
    return summary_df


def build_strategy_matched_subset(summary_df,bin_width=5,bin_range=[-50,50]):
    '''
        Sub samples from the mice to build a dataset that has similiar
        strategy index values across cre-lines

        1. First divide each cre line into bins
        2. For each bin, find the cre line with the least mice (n) in
            that bin. Sample n mice from each cre-line
    '''
    # Bin sessions into intervals
    bins = np.arange(bin_range[0],bin_range[1]+bin_width,bin_width)
    summary_df['strategy_bins'] = pd.cut(summary_df['strategy_dropout_index'],
        bins,labels=False)

    # Count how many sessions are in each bin from each cre line
    counts = summary_df.groupby(['strategy_bins','cre_line']).count()\
        .unstack(fill_value=0).stack()['behavior_session_id']

    # Iterate over bins, sample the minimum number of sessions
    # across the cre lines
    samples = []
    cres = summary_df['cre_line'].unique()
    for index, val in enumerate(bins):
        b = index+1

        # Check if there are any sessions in this bin
        if b in counts.index:
            bin_samples = counts.loc[b,:].min()

            # Check if all cre lines have at least one session 
            if bin_samples > 0:

                # Iterate cre lines and sample
                for cre in cres:
                    cre_sample = np.random.choice(summary_df.query('strategy_bins==@b')\
                        .query('cre_line==@cre').index,bin_samples,replace=False)
                    samples.append(cre_sample)

    # Record samples as strategy subset
    summary_df['strategy_matched'] = False
    summary_df.loc[np.concatenate(samples),'strategy_matched']=True

    return summary_df

def build_change_table(summary_df, version):
    ''' 
        Builds a table of all image changes in the dataset
        
        Loads the session_df for each behavior_session_id in summary_df
        Saves the change table as "_change_table.pkl"            
    '''
    # Build a dataframe for each session
    dfs = []
    crashed = []
    print('Processing Sessions')
    for index, row in tqdm(summary_df.iterrows(),total=summary_df.shape[0]):
        try:
            strategy_dir = pgt.get_directory(version, subdirectory='strategy_df')
            session_df = pd.read_csv(strategy_dir+str(row.behavior_session_id)+'.csv')
            df = session_df.query('is_change').reset_index(drop=True)
            df['behavior_session_id'] = row.behavior_session_id
            df = df.rename(columns={'image_index':'post_change_image'})
            df['pre_change_image'] = df['post_change_image'].shift(1)
            df['image_repeats'] = df['stimulus_presentations_id'].diff()
            df = df.drop(columns=['stimulus_presentations_id','image_name',
                                  'omitted','is_change','change'],errors='ignore')
            dfs.append(df)
        except Exception as e:
            crashed.append(row.behavior_session_id)

    # If any sessions crashed, print warning
    if len(crashed) > 0:
        print(str(len(crashed)) + ' sessions crashed')  
 
    print('Concatenating Sessions')
    change_df = pd.concat(dfs).reset_index(drop=True)

    print('Saving')
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    change_df.to_pickle(model_dir+'_change_table.pkl')

    return change_df, crashed

 
def get_change_table(version):
    '''
        Loads the summary change_df from file
    '''
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    return pd.read_pickle(model_dir+'_change_table.pkl')


def build_licks_table(summary_df, version):
    ''' 
        Builds a table of all image licks in the dataset
        
        Loads the session_df for each behavior_session_id in summary_df
        Saves the licks table as "_licks_table.pkl"            
    '''
    # Build a dataframe for each session
    dfs = []
    crash = 0
    crashed = []
    print('Processing Sessions')
    for index, row in tqdm(summary_df.iterrows(),total=summary_df.shape[0]):
        try:
            strategy_dir = pgt.get_directory(version, subdirectory='licks_df')
            df = pd.read_csv(strategy_dir+str(row.behavior_session_id)+'.csv')
        except Exception as e:
            crash +=1
            crashed.append(row.behavior_session_id)
        else:
            df.reset_index(drop=True)
            df = df.drop(columns=['frame'])
            df['behavior_session_id'] = row.behavior_session_id
            dfs.append(df)

    # If any sessions crashed, print warning
    if crash > 0:
        print(str(crash) + ' sessions crashed')  
    else:
        print('Loaded all sessions')
 
    print('Concatenating Sessions')
    licks_df = pd.concat(dfs).reset_index(drop=True)

    print('Saving')
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    licks_df.to_pickle(model_dir+'_licks_table.pkl')

    return licks_df, crashed 


def get_licks_table(version):
    '''
        Loads the summary licks_df from file
    '''
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    return pd.read_pickle(model_dir+'_licks_table.pkl')


def build_bout_table(licks_df):
    '''
        Generates a bouts dataframe from a lick dataframe
        Operates on either a session licks_df or summary licks_df
    
        behavior_session_id (int)
        bout_number (int) ordinal count within each session
        bout_length (int) number of licks in bout
        bout_duration (float) duration of bout in seconds
        bout_rewarded (bool) whether this bout was rewarded
        pre_ibi (float) time from the end of the last bout to 
            the start of this bout
        post_ibi (float) time until the start of the next bout
            from the end of this bout
        pre_ibi_from_start (float) time from the start of the last bout
            to the start of this bout
        post_ibi_from_start (float) time from the start of this bout
            to the start of the next

    '''

    # Groups licks into bouts
    bout_df = licks_df.groupby(['behavior_session_id',
        'bout_number']).apply(len).to_frame().rename(columns={0:"bout_length"})
    
    # count length of bouts
    bout_df['bout_duration'] = licks_df.groupby(['behavior_session_id',
        'bout_number']).last()['timestamps'] \
        - licks_df.groupby(['behavior_session_id','bout_number']).first()['timestamps']
    
    # Annotate rewarded bouts
    bout_df['bout_rewarded'] = licks_df.groupby(['behavior_session_id',
        'bout_number']).any('rewarded')['bout_rewarded']
    bout_df['bout_num_rewards'] = licks_df.groupby(['behavior_session_id',
        'bout_number']).nth(0)['bout_num_rewards']
   
    # Compute inter-bout-intervals
    bout_df['pre_ibi'] = licks_df.groupby(['behavior_session_id',
        'bout_number']).nth(0)['pre_ili']
    bout_df['post_ibi'] = licks_df.groupby(['behavior_session_id',
        'bout_number']).nth(-1)['post_ili']
    bout_df['pre_ibi_from_start'] = bout_df['pre_ibi'] \
        + bout_df['bout_duration'].shift(1)
    bout_df['post_ibi_from_start'] = bout_df['post_ibi'] \
        + bout_df['bout_duration']

    # Annotate whether the previous bout was rewarded
    bout_df['post_reward'] = bout_df['bout_rewarded'].shift(1)
    bout_df =  bout_df.reset_index()
    bout_df.loc[bout_df['bout_number']==1,'post_reward'] = False

    # Assert ibi always less than 700ms
    assert len(bout_df.query('pre_ibi < .7'))==0,\
        "Interbout interval should be less than 700ms"
    assert len(bout_df.query('post_ibi < .7'))==0,\
        "Interbout interval should be less than 700ms"

    # Check last bout of every session has NaN post_ibi
    unique_last_bout_post_ibi = \
        bout_df.groupby(['behavior_session_id']).nth(-1)['post_ibi'].unique()
    assert len(unique_last_bout_post_ibi) == 1, \
        "post_ibi for the last bout should always be NaN"
    assert np.isnan(unique_last_bout_post_ibi[0]), \
        "post_ibi for the last bout should always be NaN"  

    # Check first bout of every session has NaN pre_ibi
    unique_first_bout_pre_ibi = \
        bout_df.groupby(['behavior_session_id']).nth(0)['pre_ibi'].unique()
    assert len(unique_first_bout_pre_ibi) == 1, \
        "pre_ibi for the first bout should always be NaN"
    assert np.isnan(unique_first_bout_pre_ibi[0]), \
        "pre_ibi for the first bout should always be NaN"  

    # Check first bout of every session is not post_reward
    unique_first_bout_post_reward = \
        bout_df.groupby(['behavior_session_id']).nth(0)['post_reward'].unique()
    assert len(unique_first_bout_post_reward) == 1, \
        "post_reward for the first bout should always be False"
    assert not unique_first_bout_post_reward[0], \
        "post_reward for the first bout should always be False"  

    # Check that all rewarded licks are accounted for
    num_rewarded_licks = licks_df['num_rewards'].sum()
    num_rewarded_bouts = bout_df['bout_num_rewards'].sum()
    assert num_rewarded_licks == num_rewarded_bouts, \
        "number of rewarded licks and rewarded bouts mis-match"

    return bout_df

def build_comparison_df(df1,df2, version1,version2):
    merged_df = pd.merge(
        df1,
        df2,
        how='inner',
        on='behavior_session_id',
        suffixes=('_'+version1,'_'+version2),
        validate='1:1'
        )
    return merged_df


