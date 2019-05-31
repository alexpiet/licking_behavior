import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
plt.ion() # makes non-blocking figures

from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

# Define which experiment id you want
#ophys_experiment_id = 783927872
experiment_id = 715887471
#ophys_data, dataset, analysis,stims,fr = get_session(experiment_id)
cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'
ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir, plot_roi_validation=False)
dataset= VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)

#plt.plot(dataset.running_speed.time.values, dataset.running_speed.running_speed.values)
#plt.plot(dataset.licks.time.values, np.zeros(np.shape(dataset.licks.frame)), 'go')

# get list of all lick times
licks = dataset.licks.time.values

# get start/stop time for session
start_time = 1
stop_time = int(np.round(dataset.running_speed.time.values[-1],2)*(1/dt))

dt = 0.01 # 10msec timesteps
licks = np.round(licks,2)
licksdt = np.round(licks*(1/dt))
time_vec = np.arange(0,stop_time/100.0,dt)

# set up basic model, which has a constant lick rate
def mean_lick_model(mean_lick_rate,licksdt, stop_time):
    base = np.ones((stop_time,))*mean_lick_rate
    latent = np.exp(base)
    return loglikelihood(licksdt,latent), latent

# compute the negative log likelihood of poisson observations, given a latent vector
def loglikelihood(licksdt, latent):
    NLL = -sum(np.log(latent)[licksdt.astype(int)]) + sum(latent)
    return NLL

# Wrapper function for optimization that only takes one input
def mean_wrapper_func(mean_lick_rate):
    return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]

# optimize
res_mean = minimize(mean_wrapper_func, 1)

# We get a sensible result!
Average_probability_of_lick = np.exp(res_mean.x)[0]
sanity_check = len(licks)/(stop_time + 0.000001)

# evaluate fit by plotting prediction and lick times
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

# BIC = log(#num-data-points)*#num-params - 2*log(L)
#     = log(x)*k + 2*NLL
def compute_bic(nll, num_params, num_data_points):
    return np.log(num_data_points)*num_params + 2*nll

# make a figure
res_mean.nll, res_mean.latent = mean_lick_model(res_mean.x, licksdt, stop_time)
res_mean.BIC = compute_bic(res_mean.nll, len(res_mean.x), len(res_post_lick.latent))
compare_model(res_mean.latent, time_vec, licks, stop_time)


### Lets improve our model with a post-licking filter
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

# optimize
res_post_lick = minimize(post_lick_wrapper_func, np.ones(21,))
res_post_lick.nll,res_post_lick.latent = mean_post_lick_model(res_post_lick.x, licksdt,stop_time)
res_post_lick.BIC = compute_bic(res_post_lick.nll, len(res_post_lick.x), len(res_post_lick.latent))
compare_model(res_post_lick.latent, time_vec, licks, stop_time)

if res_post_lick.BIC < res_mean.BIC:
    print('BIC favors the post-lick filter')

# But how long of a filter should we use? We can do model optimization to find out
models = []
models.append(res_mean)
keep_going = True
current_val = 1
numbad = 0
while keep_going:
    res = minimize(post_lick_wrapper_func, np.ones(1+current_val,))
    res.nll,res.latent = mean_post_lick_model(res.x, licksdt,stop_time)
    res.BIC = compute_bic(res.nll, len(res.x), len(res.latent))   
    models.append(res)
    if models[current_val].BIC < models[current_val - 1].BIC:
        print('BIC favors extending the model '+ str(current_val)+" "+str(res.BIC))
        current_val +=1
        numbad = 0
    else:
        print('BIC does not favor extending the model, stopping')
        numbad +=1
        current_val +=1
    if numbad > 3:
        keep_going= False



# And the winner is!










