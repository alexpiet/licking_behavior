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

def build_summary_table(version):
    ''' 
        Saves out the model manifest as a csv file 
    '''
    model_manifest = ps.build_model_manifest(version=version,container_in_order=False)
    model_manifest.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D'],inplace=True)
    model_dir = ps.get_directory(version) 
    model_manifest.to_csv(model_dir+'_summary_table.csv')
    model_manifest.to_csv(OUTPUT_DIR+'_summary_table.csv')

def build_training_summary_table(version):
    ''' 
        Saves out the model manifest as a csv file 
    '''
    model_manifest = ps.build_model_training_manifest(version)
    model_manifest.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D'],inplace=True,errors='ignore') 
    model_dir = ps.get_directory(version) 
    model_manifest.to_csv(model_dir+'_training_summary_table.csv')
    model_manifest.to_csv(OUTPUT_DIR+'_training_summary_table.csv')

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
            if TRAIN:
                if not os.path.isfile(OUTPUT_DIR+str(id)+"_training.csv"):
                    build_train_session_output(id, version)
            else:
                if not os.path.isfile(OUTPUT_DIR+str(id)+".csv"):
                    build_session_output(id, version)
        except:
            num_crashed +=1
            if verbose:
                print('Session CRASHED: ' + str(id))
    print(str(num_crashed) + ' sessions crashed')
    print(str(len(ids) - num_crashed) + ' sessions saved')
    
def build_session_output(id,version):
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


################################ In development below here

def build_list_of_train_model_crashes(model_dir, try_load_data=False):
    '''
        Builds and returns a dataframe that contains information on whether a model fit is available for each 
        behavior_session_id in the training_manifest. 
        if try_load_data, will attempt to load the training data, and indicate whether data load was successful or not
    '''

    manifest = pgt.get_training_manifest().query('active').copy()
    
    for index, row in manifest.iterrows():
        try:
            fit = ps.load_fit(row.name,directory=model_dir, TRAIN=True)
            manifest.at[index,'train_model_fit']=True
        except:
            manifest.at[index,'train_model_fit']=False
            if try_load_data:
                try:
                    session = pgt.get_training_data(row.name)
                    manifest.at[index,'train_data_load'] = True
                except:
                    manifest.at[index,'train_data_load'] = False
        try:
            fit = ps.load_fit(row.name,directory=model_dir, TRAIN=False)
            manifest.at[index,'ophys_model_fit']=True
        except:
            manifest.at[index,'ophys_model_fit']=False
            if try_load_data:
                try:
                    session = pgt.get_data(row.name)
                    manifest.at[index,'ophys_data_load'] = True
                except:
                    manifest.at[index,'ophys_data_load'] = False           
    return manifest

def build_list_of_model_crashes(model_dir,try_load_data=False):
    '''
        Builds and returns a dataframe that contains information on whether a model fit is available for each 
        behavior_session_id in the manifest. 
        if try_load_data, will attempt to load the training data, and indicate whether data load was successful or not
    '''
    manifest = pgt.get_manifest().query('active').copy()
    
    for index, row in manifest.iterrows():
        try:
            fit = ps.load_fit(row.name,directory=model_dir)
            manifest.at[index,'model_fit']=True
        except:
            manifest.at[index,'model_fit']=False
            if try_load_data:
                try:
                    session = pgt.get_data(row.name)
                    manifest.at[index,'data_load'] = True
                except:
                    manifest.at[index,'data_load'] = False           
    return manifest


def build_train_session_output(id,model_dir, output_dir):
    '''
        Saves an analysis file in <output_dir> for the model fit of session <id> in <model_dir>
        Extends model weights to be constant during licking bouts
    '''   
    # Get Stimulus Info, append model free metrics
    session = pgt.get_training_data(id)
    pm.get_metrics(session,add_running=False)

    # Load Model fit
    fit = ps.load_fit(id, directory=model_dir,TRAIN=True) 
 
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
    model_output.drop(columns=['duration', 'end_frame', 'image_set','index', 'orientation', 'start_frame', 'start_time', 'stop_time', 'licks', 'rewards', 'time_from_last_lick', 'time_from_last_reward', 'time_from_last_change','bout_start', 'num_bout_start','bout_end', 'num_bout_end','change_with_lick','change_without_lick','non_change_with_lick','non_change_without_lick'],inplace=True) 

    # Save out dataframe
    model_output.to_csv(output_dir+str(id)+'_training.csv') 
    #  Read in with pd.read_csv(filename) 



