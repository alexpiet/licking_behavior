import numpy as np
import os
import fit_tools
import matplotlib.pyplot as plt
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

def get_lick_probability(model,verbose=True):
    '''
    Calculate the average probability of licking predicted by the model in time bins where the mouse licked, and time bins where the mouse did not lick
    
    Args:
        model, model object to analyze
        verbose, if True, prints a summary report in the terminal

    returns licking probability in the lick-bins, and non-lick bins
    '''
    mean_lick_prob = np.mean(model.res.latent[model.licksdt.astype(int)])
    mean_non_lick_prob = (np.sum(model.res.latent) - np.sum(model.res.latent[model.licksdt.astype(int)]))/(len(model.res.latent)-len(model.licksdt))
    if verbose:
        print('Average Licking Probability: '+str(mean_lick_prob))
        print('Average Non-Licking Probability: '+str(mean_non_lick_prob))
    return mean_lick_prob, mean_non_lick_prob

def get_mean_inter_lick_intervals(licks):
    difflicks = np.diff(licks)
    difflicks = difflicks[difflicks < .5]
    return np.mean(difflicks) 

def compare_inter_lick(experiment_id):
    model_mean_post_lick = get_peaks(experiment_id,'post_lick')
    mean_iti = get_lick_iti(experiment_id) 
    return model_mean_post_lick, mean_iti

def get_lick_iti(experiment_id):
    dt = 0.01
    data = fit_tools.get_data(experiment_id, save_dir='/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor')
    licks, licksdt, start_time, stop_time, time_vec, running_speed, rewardsdt, flashesdt, change_flashesdt,running_acceleration = fit_tools.extract_data(data,dt)
    mean_iti = get_mean_inter_lick_intervals(licks)
    return mean_iti

def compare_all_inter_licks(IDS=None,plot_this=True):
    if IDS == None:
        IDS = [837729902, 838849930,836910438,840705705,840157581,841601446,840702910,841948542,841951447,842513687,842973730,843519218,846490568,847125577,848697604]
    models = []
    datas = []
    for experiment_id in IDS:
        print(str(experiment_id))
        try:
            model_iti, data_iti = compare_inter_lick(experiment_id)
            models.append(model_iti)
            datas.append(data_iti)
        except:
            print('crash')
    if plot_this:
        plt.plot(models, datas, 'ko')
        plt.xlim(0,.25)
        plt.ylim(0,.25)
        plt.ylabel('data')
        plt.xlabel('model')
        plt.plot([0,1],[0,1], 'k--')
        plt.title('Interlick Intervals')
    return models, datas 


