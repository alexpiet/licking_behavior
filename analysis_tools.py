import numpy as np
import os
import fit_tools

def get_model(experiment_id):
    ''' 
        Internal function for loading a model object for a given session id
    '''
    fit_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/cluster_jobs'
    Fn = 'glm_model_vba_v2_'+str(experiment_id)+'.pkl'
    full_path = os.path.join(fit_path, Fn)
    model = fit_tools.Model.from_file_rebuild(full_path)
    return model

def get_peaks(experiment_id,filter_name):
    '''
        Returns the peak time of the given filter in seconds

        Args:
        experiment_id, id of the session to analyze
        filter_name, name of the filter to get the peak of
        
        Accepted filter names are: 
        post_lick
        running_speed
        reward
        flash
        change_flash
        running_acceleration

        Returns the peak time in seconds
    '''
    # loads model
    model = get_model(experiment_id)

    # gets filters
    f = model.linear_filter(filter_name)

    # gets peaks
    peak_time = np.argmax(f)

    # returns
    return peak_time*model.dt




