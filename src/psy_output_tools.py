import psy_general_tools as pgt
import psy_metrics_tools as pm
import psy_tools as ps
import numpy as np
import os

def build_id_fit_list(VERSION):
    '''
        Saves out two text files with lists of all behavior_session_ids for ophys and training sessions in the manifest
        Only include active sessions
    '''
    # Get manifest
    manifest = pgt.get_manifest()
    training = pgt.get_training_manifest()
    
    # Set filenames
    fname = '/home/alex.piet/codebase/behavior/licking_behavior/scripts/psy_ids_v'+str(VERSION)+'.txt'
    ftname ='/home/alex.piet/codebase/behavior/licking_behavior/scripts/psy_training_ids_v'+str(VERSION)+'.txt'

    # Filter and save
    np.savetxt(fname,  manifest.query('active').index.values)
    np.savetxt(ftname, training.query('active').index.values)

def build_list_of_train_model_crashes(model_dir):
    manifest = pgt.get_training_manifest().query('active').copy()
    
    for index, row in manifest.iterrows():
        try:
            fit = ps.load_fit(row.name,directory=model_dir, TRAIN=True)
            manifest.at[index,'train_model_fit']=True
        except:
            manifest.at[index,'train_model_fit']=False
        try:
            fit = ps.load_fit(row.name,directory=model_dir, TRAIN=False)
            manifest.at[index,'ophys_model_fit']=True
        except:
            manifest.at[index,'ophys_model_fit']=False
    return manifest

def build_list_of_model_crashes(model_dir):
    manifest = pgt.get_manifest().query('active').copy()
    
    for index, row in manifest.iterrows():
        try:
            fit = ps.load_fit(row.name,directory=model_dir)
            manifest.at[index,'model_fit']=True
        except:
            manifest.at[index,'model_fit']=False
    return manifest



def build_summary_table(model_dir,output_dir):
    ''' 
        Saves out the model manifest as a csv file 
    '''
    model_manifest = ps.build_model_manifest(directory=model_dir,container_in_order=False)
    model_manifest.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D'],inplace=True) 
    model_manifest.to_csv(output_dir+'_summary_table.csv')

def build_training_summary_table(model_dir,output_dir,hit_threshold=10):
    ''' 
        Saves out the model manifest as a csv file 
    '''
    model_manifest = ps.build_model_training_manifest(directory=model_dir,hit_threshold=10)
    model_manifest.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D'],inplace=True,errors='ignore') 
    model_manifest.to_csv(output_dir+'_training_summary_table.csv')

def build_all_session_outputs(ids,model_dir,output_dir,TRAIN=False):
    '''
        Iterates a list of session ids, and builds the results file. 
        If TRAIN, uses the training interface
    '''
    # Iterate each session
    for index, id in enumerate(ids):
        print(index)
        try:
            if TRAIN:
                if not os.path.isfile(output_dir+str(id)+"_training.csv"):
                    build_train_session_output(id, model_dir, output_dir)
                print('Training Session done: ' + str(id))
            else:
                if not os.path.isfile(output_dir+str(id)+".csv"):
                    build_session_output(id, model_dir, output_dir)
                print('Session done: ' + str(id))
        except:
            print('Session CRASHED: ' + str(id))

def build_session_output(id,model_dir, output_dir):
    '''
        Saves an analysis file in <output_dir> for the model fit of session <id> in <model_dir>
        Extends model weights to be constant during licking bouts
    '''
    # Get Stimulus Info, append model free metrics
    session = pgt.get_data(id)
    pm.get_metrics(session)

    # Load Model fit
    fit = ps.load_fit(id, directory=model_dir)
 
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
    model_output.drop(columns=['duration', 'end_frame', 'image_set','index', 'orientation', 'start_frame', 'start_time', 'stop_time', 'licks', 'rewards', 'time_from_last_lick', 'time_from_last_reward', 'time_from_last_change', 'mean_running_speed', 'bout_start', 'num_bout_start','bout_end', 'num_bout_end','change_with_lick','change_without_lick','non_change_with_lick','non_change_without_lick'],inplace=True) 

    # Save out dataframe
    model_output.to_csv(output_dir+str(id)+'.csv') 
    #  Read in with pd.read_csv(filename) 


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



