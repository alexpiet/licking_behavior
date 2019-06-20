import numpy as np
import os
import fit_tools
import matplotlib.pyplot as plt
import filters
import new_model_obj as mo

def get_old_model(experiment_id):
    ''' 
        Internal function for loading a model object for a given session id
    '''
    #### OLD MODEL OBJECT
    fit_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/cluster_jobs'
    Fn = 'glm_model_vba_v2_'+str(experiment_id)+'.pkl'
    full_path = os.path.join(fit_path, Fn)
    model = fit_tools.Model.from_file_rebuild(full_path)
    return model

def get_model(experiment_id):
    #### LOADS A SPECIFIC MODEL ITERATION, NOT STABLE
    output_dir = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_jobs/20190614_glm_fit'
    param_save_fn = 'model_{}_71_saved_params.npz'.format(experiment_id)
    param_save_full_path = os.path.join(output_dir, param_save_fn)
    dt = 0.01
    data = fit_tools.get_data(experiment_id, save_dir='/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor')
    (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
     running_speed, running_timestamps, running_acceleration, timebase,
     time_start, time_end) = mo.bin_data(data, dt)
    model = mo.Model(dt=0.01,
                  licks=licks_vec, 
                  verbose=True,
                  name='{}'.format(experiment_id),
                  l2=0.5)
    post_lick_filter = mo.GaussianBasisFilter(data = licks_vec,
                                           dt = model.dt,
                                           **filters.post_lick)
    model.add_filter('post_lick', post_lick_filter)
    reward_filter = mo.GaussianBasisFilter(data = rewards_vec,
                                        dt = model.dt,
                                        **filters.reward)
    model.add_filter('reward', reward_filter)
    flash_filter = mo.GaussianBasisFilter(data = flashes_vec,
                                       dt = model.dt,
                                       **filters.flash)
    model.add_filter('flash', flash_filter)
    change_filter = mo.GaussianBasisFilter(data = change_flashes_vec,
                                        dt = model.dt,
                                        **filters.change)
    model.add_filter('change_flash', change_filter)
    running_speed_filter= mo.GaussianBasisFilter(data = running_speed,
                                              dt = model.dt,
                                              **filters.running_speed)
    model.add_filter('running_speed', running_speed_filter)
    acceleration_filter= mo.GaussianBasisFilter(data = running_acceleration,
                                             dt = model.dt,**filters.acceleration)
    model.add_filter('acceleration', acceleration_filter)
    model.set_filter_params_from_file(param_save_full_path)
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
    my_f = model.filters[filter_name]
    f,basis = my_f.build_filter()
    #f = model.linear_filter(filter_name) #### OLD MODEL OBJECT

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
    nll,latent = model.calculate_latent()
    licksdt = np.flatnonzero(model.licks)
    mean_lick_prob = np.mean(latent[latent.astype(int)])
    mean_non_lick_prob = (np.sum(latent) - np.sum(latent[licksdt.astype(int)]))/(len(latent)-len(licksdt))
 
    #### OLD MODEL OBJECT
    #mean_lick_prob = np.mean(model.res.latent[model.licksdt.astype(int)])
    #mean_non_lick_prob = (np.sum(model.res.latent) - np.sum(model.res.latent[model.licksdt.astype(int)]))/(len(model.res.latent)-len(model.licksdt))
    if verbose:
        print('Average Licking Probability: '+str(mean_lick_prob))
        print('Average Non-Licking Probability: '+str(mean_non_lick_prob))
    return mean_lick_prob, mean_non_lick_prob

def get_mean_inter_lick_intervals(licks):
    ''' 
        Calculates the mean inter lick interval
    
    Args:
        licks, a list of the lick times
    '''
    difflicks = np.diff(licks)
    difflicks = difflicks[difflicks < .25]
    return np.mean(difflicks) 

def compare_inter_lick(experiment_id):
    '''
        computes the model predicted ili, and the data ili
    Args:
        experiment_id, the session to run
    returns
        model_mean_post_lick, the time of the peak of the post-lick filter
        mean_ili, the data mean of the ili
    '''
    model_mean_post_lick = get_peaks(experiment_id,'post_lick')
    mean_ili = get_lick_ili(experiment_id) 
    return model_mean_post_lick, mean_ili

def get_lick_ili(experiment_id):
    '''
        gets the data ili for <experiment_id>
    '''
    dt = 0.01
    data = fit_tools.get_data(experiment_id, save_dir='/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor')
    licks, licksdt, start_time, stop_time, time_vec, running_speed, rewardsdt, flashesdt, change_flashesdt,running_acceleration = fit_tools.extract_data(data,dt)
    mean_ili = get_mean_inter_lick_intervals(licks)
    return mean_ili

def compare_all_inter_licks(IDS=None,plot_this=True):
    '''
        Computes the data and model ili for all sessions in <IDS>. Plots a scatter plot
    '''
    if IDS == None:
        IDS = [837729902, 838849930,836910438,840705705,840157581,841601446,840702910,841948542,841951447,842513687,842973730,843519218,846490568,847125577,848697604]
    models = []
    datas = []
    for experiment_id in IDS:
        print(str(experiment_id))
        try:
            model_ili, data_ili = compare_inter_lick(experiment_id)
            models.append(model_ili)
            datas.append(data_ili)
        except:
            print('crash')
    if plot_this:
        plt.figure()
        plt.plot(models, datas, 'ko',alpha = .3)
        plt.xlim(0.1,.2)
        plt.ylim(0.1,.2)
        plt.ylabel('Average ILI in Data (s)',fontsize=20)
        plt.xlabel('Average ILI in Model (s)',fontsize=20)
        plt.plot([0,1],[0,1], 'k--',alpha = .2)
        plt.title('Interlick Intervals',fontsize=24)
        ax = plt.gca()
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
    return models, datas 

def compare_dist(IDS=None, plot_this=True,variable='licks'):
    '''
        Plots the event triggered average lick rates for the data, and overlays the learned filter for that event
        IDS: session ids to analyze (each get plotted separatel)
        plot_this: plot the results, or just return the values
        variable: which task events to analyze. Must be one of ('licks', 'rewards', 'flash', 'change_flash')
    '''
    if IDS == None:
        IDS = [837729902, 838849930,836910438,840705705,840157581,841601446,840702910,841948542,841951447,842513687,842973730,843519218,846490568,847125577,848697604]
    if not ((variable == 'licks') | (variable == 'rewards') | (variable =='flash') | (variable == 'change_flash') | (variable == 'running_speed') | (variable == 'running_acceleration')):
        raise Exception('Unknown variable')
    for experiment_id in IDS:
        print(str(experiment_id))
        try:
            times,numbins = get_dist(experiment_id, variable) 
            my_filter,time_vec = get_filter(experiment_id, variable)
        except:
            print('   crash')
        else:
            if plot_this:
                plt.figure()
                ax = plt.gca()
                ax2 = ax.twinx()
                if ((variable == 'licks') | (variable == 'rewards') | (variable =='flash') | (variable == 'change_flash') ):
                    ax.hist(times,bins=numbins)               
                    ax.set_ylabel('lick density',color='b')
                    ax.set_xlabel('time from '+variable +' (s)')
                    ax2.plot(time_vec,my_filter,'k',linewidth=2)
                    ax2.set_ylim(ymin=0)
                else:
                    ax.plot(numbins, times,'b',linewidth=2)
                    ax.set_ylabel('Lick Triggered Average ' + variable) 
                    ax.set_xlabel('time from lick (s)')
                    ax2.plot(time_vec,my_filter,'k',linewidth=2)
                plt.title(variable)
                ax2.set_ylabel('filter gain', color='k')
                plt.tight_layout()
    return 

def get_dist(experiment_id, variable):
    '''
        gets the event triggered average lick rate in the data
    '''
    dt = 0.01
    data = fit_tools.get_data(experiment_id, save_dir='/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor')
    licks, licksdt, start_time, stop_time, time_vec, running_speed, rewardsdt, flashesdt, change_flashesdt,running_acceleration = fit_tools.extract_data(data,dt)
    if len(licks) < 5:
        raise Exception(' bad session')
    if len(rewardsdt) < 5:
        raise Exception(' bad session')
    if variable == 'licks':
        difflicks = np.diff(licks)
        difflicks = difflicks[difflicks < .5]
        return difflicks,20
    elif variable == 'flash':
        triggered = trigger_on_licks(licks, flashesdt*dt)
        triggered = triggered[triggered< .750]
        return triggered,15
    elif variable == 'change_flash':
        triggered = trigger_on_licks(licks, change_flashesdt*dt)
        triggered = triggered[triggered< 1.50]
        return triggered,20
    elif variable == 'rewards':
        triggered = trigger_on_licks(licks, rewardsdt*dt)
        triggered = triggered[triggered < 4]
        return triggered,30
    elif variable == 'running_speed':
        triggered,time_vec = get_lick_triggered_average(licksdt, running_speed, 100)
        return triggered,time_vec*dt
    elif variable == 'running_acceleration':
        triggered,time_vec = get_lick_triggered_average(licksdt, running_acceleration, 100)
        return triggered, time_vec*dt
    else:
        raise Exception('Unknown variable') 

def get_lick_triggered_average(licksdt, series,numbins,direction='backwards'):
    '''
        Computes the lick triggered average for a time-series

        ARGS:
        licksdt, the time-bin index of licks
        series, a time series of events
        numbins, the number of timebins away from a lick to compute the LTA on
        direction, do you want the LTA either forward, or backwards (causal) to the link?

        RETURNS:
        The LTA, and the time basis for that LTA
    '''
    all_series = np.zeros((len(licksdt), numbins))
    all_series[:] = 0
    for i in np.arange(0,len(licksdt)):
        if direction == 'backwards':
            my_series = series[int(licksdt[i])-numbins:int(licksdt[i])]
        elif direction == 'forwards':
            raise Exception('need to implement this')
        all_series[i,:]=my_series
    if direction == 'backwards':
        time_vec = np.arange(-numbins,0)
    return np.mean(all_series, 0), time_vec

def get_filter(experiment_id, variable):
    '''
        Returns the time series value of the exponentiated filter for session <experiment_id> for event <variable>
        Returns:
        my_filter       the filter time series
        time_vec        the time values for my_filter
    '''
    model = get_model(experiment_id)
    filter_name = variable
    if variable == 'licks':
        filter_name = 'post_lick'
    if variable == 'rewards':
        filter_name = 'reward'
    if variable == 'running_acceleration':
        filter_name = 'acceleration'
    #    f = model.linear_filter(filter_name) #### OLD MODEL OBJECT
    my_f = model.filters[filter_name]
    f,basisfuncs = my_f.build_filter()
    dt = model.dt
    time_vec = np.arange(0, len(f))*dt
    if (variable == 'running_speed' ) | (variable == 'running_acceleration'):
        time_vec = np.arange(-len(f),0)*dt
    my_filter =np.exp(f)
    return my_filter,time_vec

def trigger_on_licks(licks,times):
    '''
        aligns the lick times in <licks> to the event times in <times>

        ARGS:   
        licks, vector of lick times
        times, vector event times

        RETURNS: 
        the times of all licks relative to the most immediately proceeding event in times
    '''
    triggered = np.array([])
    for i in np.arange(0,len(times)):
        if i < len(times)-1:
            mylicks = licks[(licks >=times[i]) & (licks < times[i+1])] - times[i]
        else:
            mylicks = licks[(licks >=times[i])] - times[i]
        triggered = np.concatenate([triggered, mylicks])
    return triggered




