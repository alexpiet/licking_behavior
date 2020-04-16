import psy_general_tools as pgt
import psy_metrics_tools as pm
import psy_tools as ps
import numpy as np

def build_id_fit_list(VERSION):
    manifest = pgt.get_manifest()
    training = pgt.get_training_manifest()
    
    fname = '/home/alex.piet/codebase/behavior/licking_behavior/scripts/psy_ids_v'+str(VERSION)+'.txt'
    ftname ='/home/alex.piet/codebase/behavior/licking_behavior/scripts/psy_training_ids_v'+str(VERSION)+'.txt'

    np.savetxt(fname, manifest.index.values)
    np.savetxt(ftname, training.index.values)


def build_summary_table(model_dir,output_dir):
   model_manifest = ps.build_model_manifest(directory=model_dir,container_in_order=False)
   model_manifest.drop(columns=['weight_bias','weight_omissions1','weight_task0','weight_timing1D'],inplace=True) 
   model_manifest.to_csv(output_dir+'_summary_table.csv')

def build_all_session_outputs(ids,model_dir,output_dir):
   # Iterate each session
   for index, id in enumerate(ids):
       print(index)
       try:
           build_session_output(id, model_dir, output_dir)
       except:
           print('crashed: ' + str(id))

def build_session_output(id,model_dir, output_dir):
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




