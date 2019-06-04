# Import packages
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

## This code base expects some variables with the following formats
## get list of all lick times, an array with the lick times, rounded to the nearest 10msec
#licks = dataset.licks.time.values
#dt = 0.01 # 10msec timesteps
#licks = np.round(licks,2)
#licksdt = np.round(licks*(1/dt))
## get start/stop time for session
#start_time = 1
#stop_time = int(np.round(dataset.running_speed.time.values[-1],2)*(1/dt))
## A time vector in dt increments from 0 to stop_time
# time_vec = np.arange(0,stop_time/100.0,dt)

#### General Functions
def loglikelihood(licksdt, latent,params=[],l2=0):
    '''
    Compute the negative log likelihood of poisson observations, given a latent vector

    Args:
        licksdt: a vector of lick times in dt-index points
        latent: a vector of the estimated lick rate in each time bin
        params: a vector of the parameters for the model
        l2: amplitude of L2 regularization penalty
    
    Returns: NLL of the model
    '''
    # If there are any zeros in the latent model, have to add "machine tolerance"
    latent[latent==0] += np.finfo(float).eps

    NLL = -sum(np.log(latent)[licksdt.astype(int)]) + sum(latent) + l2*np.sum(np.array(params)**2)
    return NLL

def compare_model(latent, time_vec, licks, stop_time, running_speed=None):
    '''
    Evaluate fit by plotting prediction and lick times

    Args:
        Latent: a vector of the estimate lick rate
        time_vec: the timestamp for each time bin
        licks: the time of each lick in dt-rounded timevalues
        stop_time: the number of timebins
    
    Plots the lick raster, and latent rate
    
    Returns: the figure handle and axis handle
    '''
    fig,ax  = plt.subplots()    
    plt.plot(time_vec,latent,'b',label='model')
    plt.vlines(licks,0, 1, alpha = 0.3, label='licks')
    if running_speed is not None:
        plt.plot(time_vec, running_speed[:-1] / np.max(running_speed), 'r-')
    plt.ylim([0, 1])
    plt.xlim(600,660)
    plt.legend(loc=9 )
    plt.xlabel('time (s)')
    plt.ylabel('Licking Probability')
    plt.tight_layout()
    return fig, ax

def compute_bic(nll, num_params, num_data_points):
    '''
    Computes the BIC of the model
    BIC = log(#num-data-points)*#num-params - 2*log(L)
        = log(x)*k + 2*NLL

    Args:
        nll: negative log likelihood of the model
        num_params: number of parameters in the model
        num_data_points: number of data points in model
    
    Returns the BIC score
    '''
    return np.log(num_data_points)*num_params + 2*nll

def evaluate_model(res,model_func, licksdt, stop_time):
    '''
    Evaluates the model

    Args:
        res: the optimization results from minimize()
        model_func: the function handle for the model
        licksdt: the lick times in dt-index
        stop_time: number of time bins

    Returns: res, with nll computed, latent estimate computed, BIC computed
    '''
    res.nll, res.latent = model_func(res.x)
    res.BIC = compute_bic(res.nll, len(res.x), len(res.latent))
    return res    

def build_filter(params,filter_time_vec, sigma, plot_filters=False, plot_nonlinear=False):
    '''
    Builds a filter out of basis functions

    puts len(params) gaussian bumps equally spaced across time_vec
    each gaussian is weighted by params, and is truncated outside of time_vec

    Args:
        params: The weights of each gaussian bumps
        filter_time_vec: the time vector of the timepoints to build the filter for
        sigma: the variance of each gaussian bump
        plot_filters: if True, plots each bump, and the entire function
    
    
    Returns: The filter, with length given by filter_time_vec
    
    Example:

    filter_time_vec = np.arange(dt,.21,dt)
    build_filter([2.75,-2,-2,-2,-2,3,3,3,.1], filter_time_vec, 0.025, plot_filters=True)
    '''
    def gaussian_template(mu,sigma):
        return (1/(np.sqrt(2*3.14*sigma**2)))*np.exp(-(filter_time_vec-mu)**2/(2*sigma**2))
    numparams = len(params)
    mean = (filter_time_vec[-1] - filter_time_vec[0])/(numparams-1)
    base = np.zeros(np.shape(filter_time_vec)) 
    if plot_filters:
        plt.figure()
    for i in range(0,len(params)):
        base += params[i]*gaussian_template(mean*i,sigma)    
        if plot_filters:
            plt.plot(filter_time_vec, params[i]*gaussian_template(mean*i,sigma))
    if plot_filters:
        plt.plot(filter_time_vec,base, 'k')
        if plot_nonlinear:
            plt.figure()
            plt.plot(filter_time_vec, np.exp(base), 'k')
    return base

def get_data(experiment_id, save_dir=r'/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor'):

    '''
    Pull processed data.

    Args: 
        experiment_id (int): The experiment ID to get data for
        save_dir (str): dir containing processed NPZ files

    Returns: 
        data (dict of np.arr): A dictionary with arrays under the following keys: 

        running_timestamps = data['running_timestamps']
        running_speed = data['running_speed']
        lick_timestamps = data['lick_timestamps']
        stim_on_timestamps = data['stim_on_timestamps']
        stim_off_timestamps = data['stim_off_timestamps']
        stim_name = data['stim_name']
        stim_omitted = data['stim_omitted']
        reward_timestamps = data['reward_timestamps']

    '''
    output_fn = 'experiment_{}.npz'.format(experiment_id)
    full_path = os.path.join(save_dir, output_fn)
    data = np.load(full_path)
    return data

#### Specific Model Functions
# set up basic model, which has a constant lick rate
# mean_lick rate: scalar parameter that is the log(average-lick rate)
# licksdt: a vector of lick times in dt-index points
# stop_time: The index of the last time-bin
#
# Returns: the NLL of the model, and the latent rate
def mean_lick_model(mean_lick_rate,licksdt, stop_time):
    '''
    Depreciated, use licking model
    '''
    base = np.ones((stop_time,))*mean_lick_rate
    latent = np.exp(base)
    return loglikelihood(licksdt,latent), latent

# Wrapper function for optimization that only takes one input
def mean_wrapper_func(mean_lick_rate):
    '''
    Depreciated, use licking model
    '''
    return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]

# Model with Mean lick rate, and post-lick filter
# params[0]: mean lick rate
# params[1:]: post-lick filter
def mean_post_lick_model(params, licksdt,stop_time):
    '''
    Depreciated, use licking model
    '''
    mean_lick_rate = params[0]
    base = np.ones((stop_time,))*mean_lick_rate
    post_lick_filter = params[1:]
    post_lick = np.zeros((stop_time+len(post_lick_filter)+1,))
    for i in licksdt:
        post_lick[int(i)+1:int(i)+1+len(post_lick_filter)] +=post_lick_filter
    post_lick = post_lick[0:stop_time]
    latent = np.exp(base+post_lick)
    return loglikelihood(licksdt,latent), latent

def post_lick_wrapper_func(params):
    '''
    Depreciated, use licking model
    '''
    return mean_post_lick_model(params,licksdt,stop_time)[0]

# Model with Mean lick rate, and post-lick filter
# params[0]: mean lick rate
# params[1:]: post-lick filter parameters for basis function
def basis_post_lick_model(params, licksdt,stop_time,sigma):
    '''
    Depreciated, use licking model
    '''
    mean_lick_rate = params[0]
    base = np.ones((stop_time,))*mean_lick_rate
    filter_time_vec = np.arange(dt,.21,dt)
    post_lick_filter = build_filter(params[1:],filter_time_vec,sigma)
    post_lick = np.zeros((stop_time+len(post_lick_filter)+1,))
    for i in licksdt:
        post_lick[int(i)+1:int(i)+1+len(post_lick_filter)] +=post_lick_filter
    post_lick = post_lick[0:stop_time]
    latent = np.exp(base+post_lick)
    return loglikelihood(licksdt,latent), latent

def basis_post_lick_wrapper_func(params):
    '''
    Depreciated, use licking model
    '''
    return basis_post_lick_model(params,licksdt,stop_time,0.025)[0]


def licking_model(params, licksdt, stop_time, mean_lick_rate=True, dt = 0.01,
    post_lick=True,num_post_lick_params=10,post_lick_duration=.21, post_lick_sigma =0.025, 
    include_running_speed=False, num_running_speed_params=10,running_speed_duration = 0.25, running_speed_sigma = 0.025,running_speed=0,
    include_reward=False, num_reward_params=10,reward_duration =4, reward_sigma = 0.5 ,rewardsdt=[],
    include_flashes=False, num_flash_params=10,flash_duration=0.750, flash_sigma = 0.025, flashesdt=[],
    include_change_flashes=False, num_change_flash_params=10,change_flash_duration=0.750, change_flash_sigma = 0.025, change_flashesdt=[],
    l2=0):
    '''
    Top function for fitting licking model. Can flexibly add new features
    
    Args:
        params,         vector of parameters
        licksdt,        dt-index of each lick time
        stop_time,      number of timebins
        mean_lick_rate, if True, include mean lick rate
        dt,             length of timestep
        For each feature:
        <feature>               if True, include this features
        num_<feature>_params    number of parameters for this feature
        <feature>_duration      length of the filter for this feature
        <feature>_sigma         width of each basis function for this feature

        l2,             penalty strength of L2 (Ridge) Regularization
    Returns:
        NLL for this model
        latent lick rate for this model
    '''
    base = np.zeros((stop_time,))
    param_counter = 0
    if mean_lick_rate:
        mean_lick_param = params[param_counter]
        param_counter +=1
        base += np.ones((stop_time,))*mean_lick_param
    if post_lick:
        param_counter, post_lick_params = extract_params(params, param_counter, num_post_lick_params)
        post_lick_response = linear_post_lick(post_lick_params,post_lick_duration,licksdt,dt,post_lick_sigma,stop_time)
        base += post_lick_response
    if include_running_speed:
        param_counter, running_speed_params = extract_params(params, param_counter, num_running_speed_params)
        running_speed_response = linear_running_speed(running_speed_params, running_speed_duration, running_speed, dt, running_speed_sigma, stop_time)
        base += running_speed_response
    if include_reward:
        param_counter, reward_params = extract_params(params, param_counter, num_reward_params)
        reward_response = linear_reward(reward_params, reward_duration, rewardsdt, dt, reward_sigma, stop_time)
        base += reward_response
    if include_flashes:
        param_counter, flash_params = extract_params(params, param_counter, num_flash_params)
        flash_response = linear_reward(flash_params, flash_duration, flashesdt, dt, flash_sigma, stop_time)
        base += flash_response
    if include_change_flashes:
        param_counter, change_flash_params = extract_params(params, param_counter, num_change_flash_params)
        change_flash_response = linear_reward(change_flash_params, change_flash_duration, change_flashesdt, dt, change_flash_sigma, stop_time)
        base += change_flash_response
    if not (param_counter == len(params)):
        print(str(param_counter))
        print(str(len(params)))
        raise Exception('Not all parameters were used')

    # Clip to prevent overflow errors
    latent = np.exp(np.clip(base, -700, 700))
    return loglikelihood(licksdt,latent,params=params, l2=l2), latent

def extract_params(params, param_counter, num_to_extract):
    '''
    Extracts each feature's parameters from the vector of model parameters

    Args:
        params      the vector of all parameters
        param_counter, the current location in the parameter list
        num_to_extract, the number of parameters for this feature
    '''
    this_params = params[param_counter:param_counter+num_to_extract]
    param_counter += num_to_extract
    if not (len(this_params) == num_to_extract):
        raise Exception('Parameter mis-alignment')
    return param_counter, this_params

def linear_post_lick(post_lick_params, post_lick_duration, licksdt,dt,post_lick_sigma,stop_time):
    '''
    Computes the linear response function for the post-lick-triggered filter

    Args:
        post_lick_params,       vector of parameters, number of parameters determines number of basis functions
        post_lick_duration,     duration (s) of the filter
        licksdt,                times of the licks in dt-index units
        post_lick_sigma,        sigma parameter for basis functions
        stop_time,              number of timebins
    '''
    filter_time_vec = np.arange(dt,post_lick_duration,dt)
    post_lick_filter = build_filter(post_lick_params,filter_time_vec,post_lick_sigma)
    post_lick = np.zeros((stop_time+len(post_lick_filter)+1,))
    for i in licksdt:
        post_lick[int(i)+1:int(i)+1+len(post_lick_filter)] +=post_lick_filter
    post_lick = post_lick[0:stop_time]       
    return post_lick

def linear_running_speed(running_speed_params, running_speed_duration, running_speed, dt, running_speed_sigma, stop_time):
    '''
    Args:
        running_speed_params (np.array): Array of parameters
        running_speed_duration (int): Length of the running speed filter in seconds
        running_speed (np.array): Actual running speed values
        dt (float): length of the time bin in seconds
        running_speed_sigma (float): standard deviation of each Gaussian basis function to use in the filter
        stop_time (int): end bin number

    Returns:
        running_effect (np.array): The effect on licking from the previous running at each time point
    '''

    filter_time_vec = np.arange(dt, running_speed_duration, dt)
    #  running_speed_filter = build_filter(running_speed_params, filter_time_vec, running_speed_sigma)
    running_speed_filter = running_speed_params
    running_effect = np.convolve(np.concatenate([np.zeros(len(running_speed_filter)), running_speed]), running_speed_filter)[:stop_time]
    
    # Shift our predictions to the next time bin
    running_effect = np.r_[0, running_effect[1:]]
    return running_effect


def linear_reward(reward_params, reward_duration, rewardsdt, dt, reward_sigma, stop_time):
    '''
    Computes the linear response function for the reward-triggered filter

    Args:
        reward_params,    vector of parameters, number of parameters determines number of basis functions
        reward_duration,  duration (s) of the filter
        rewardsdt,       times of the rewards in dt-index units
        reward_sigma,     sigma parameter for basis functions
        stop_time,              number of timebins
    '''
    filter_time_vec =np.arange(dt, reward_duration,dt)
    reward_filter = build_filter(reward_params, filter_time_vec, reward_sigma)
    base = np.zeros((stop_time+len(reward_filter)+1,))
    for i in rewardsdt:
        base[int(i)+1:int(i)+1+len(reward_filter)] += reward_filter
    base = base[0:stop_time]
    return base

def linear_flash(flash_params, flash_duration, flashesdt, dt, flash_sigma, stop_time):
    '''
    Computes the linear response function for the image-triggered filter

    Args:
        flash_params,    vector of parameters, number of parameters determines number of basis functions
        flash_duration,  duration (s) of the filter
        flashesdt,       times of the flashes in dt-index units
        flash_sigma,     sigma parameter for basis functions
        stop_time,              number of timebins
    '''
    filter_time_vec =np.arange(dt, flash_duration,dt)
    flash_filter = build_filter(flash_params, filter_time_vec, flash_sigma)
    base = np.zeros((stop_time+len(flash_filter)+1,))
    for i in flashesdt:
        base[int(i)+1:int(i)+1+len(flash_filter)] += flash_filter
    base = base[0:stop_time]
    return base

def linear_change_flash(change_flash_params, change_flash_duration, change_flashesdt, dt, change_flash_sigma, stop_time):
    '''
    Computes the linear response function for the change-image-triggered filter

    Args:
        change_flash_params,    vector of parameters, number of parameters determines number of basis functions
        change_flash_duration,  duration (s) of the filter
        change_flashesdt,       times of the change flashes in dt-index units
        change_flash_sigma,     sigma parameter for basis functions
        stop_time,              number of timebins
    '''
    filter_time_vec =np.arange(dt, change_flash_duration,dt)
    change_flash_filter = build_filter(change_flash_params, filter_time_vec, change_flash_sigma)
    base = np.zeros((stop_time+len(change_flash_filter)+1,))
    for i in change_flashesdt:
        base[int(i)+1:int(i)+1+len(change_flash_filter)] += change_flash_filter
    base = base[0:stop_time]
    return base




