# Import packages
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
# compute the negative log likelihood of poisson observations, given a latent vector
# licksdt: a vector of lick times in dt-index points
# latent: a vector of the estimated lick rate in each time bin
#
# Returns: NLL of the model
def loglikelihood(licksdt, latent):
    NLL = -sum(np.log(latent)[licksdt.astype(int)]) + sum(latent)
    return NLL

# evaluate fit by plotting prediction and lick times
# Latent: a vector of the estimate lick rate
# time_vec: the timestamp for each time bin
# licks: the time of each lick in dt-rounded timevalues
# stop_time: the number of timebins
#
# Plots the lick raster, and latent rate
#
# Returns: the figure handle and axis handle
def compare_model(latent, time_vec, licks, stop_time):
    fig,ax  = plt.subplots()    
    plt.plot(time_vec,latent,'b',label='model')
    plt.vlines(licks,0, 1, alpha = 0.3, label='licks')
    plt.ylim([0, 1])
    plt.xlim(600,660)
    plt.legend(loc=9 )
    plt.xlabel('time (s)')
    plt.ylabel('Licking Probability')
    plt.tight_layout()
    return fig, ax

# Computes the BIC of the model
# BIC = log(#num-data-points)*#num-params - 2*log(L)
#     = log(x)*k + 2*NLL
# nll: negative log likelihood of the model
# num_params: number of parameters in the model
# num_data_points: number of data points in model
#
# Returns the BIC score
def compute_bic(nll, num_params, num_data_points):
    return np.log(num_data_points)*num_params + 2*nll

# Evaluates the model
# res: the optimization results from minimize()
# model_func: the function handle for the model
# licksdt: the lick times in dt-index
# stop_time: number of time bins
#
# Returns: res, with nll computed, latent estimate computed, BIC computed
def evaluate_model(res,model_func, licksdt, stop_time):
    res.nll, res.latent = model_func(res.x, licksdt, stop_time)
    res.BIC = compute_bic(res.nll, len(res.x), len(res.latent))
    return res    

# Builds a filter out of basis functions
# params: The weights of each gaussian bumps
# filter_time_vec: the time vector of the timepoints to build the filter for
# sigma: the variance of each gaussian bump
# plot_filters: if True, plots each bump, and the entire function
#
# puts len(params) gaussian bumps equally spaced across time_vec
# each gaussian is weighted by params, and is truncated outside of time_vec
# 
# Returns: The filter, with length given by filter_time_vec
#
# Example:
# filter_time_vec = np.arange(dt,.21,dt)
# build_filter([2.75,-2,-2,-2,-2,3,3,3,.1], filter_time_vec, 0.025, plot_filters=True)
def build_filter(params,filter_time_vec, sigma, plot_filters=False):
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
    return base



#### Specific Model Functions
# set up basic model, which has a constant lick rate
# mean_lick rate: scalar parameter that is the log(average-lick rate)
# licksdt: a vector of lick times in dt-index points
# stop_time: The index of the last time-bin
#
# Returns: the NLL of the model, and the latent rate
def mean_lick_model(mean_lick_rate,licksdt, stop_time):
    base = np.ones((stop_time,))*mean_lick_rate
    latent = np.exp(base)
    return loglikelihood(licksdt,latent), latent

# Wrapper function for optimization that only takes one input
def mean_wrapper_func(mean_lick_rate):
    return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]

# Model with Mean lick rate, and post-lick filter
# params[0]: mean lick rate
# params[1:]: post-lick filter
def mean_post_lick_model(params, licksdt,stop_time):
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
    return mean_post_lick_model(params,licksdt,stop_time)[0]

# Model with Mean lick rate, and post-lick filter
# params[0]: mean lick rate
# params[1:]: post-lick filter parameters for basis function
def basis_post_lick_model(params, licksdt,stop_time,sigma):
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
    return basis_post_lick_model(params,licksdt,stop_time,0.025)[0]


