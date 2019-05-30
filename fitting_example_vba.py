import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

plt.ion() # makes non-blocking figures
from alex_utils import whos
from vba_utils import get_session

from visual_behavior.ophys.io.convert_level_1_to_level_2 import convert_level_1_to_level_2
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis

# Define which experiment id you want
#ophys_experiment_id = 783927872
ophys_experiment_id = 715887471
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
def wrapper_func(mean_lick_rate):
    return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]

# optimize
res = minimize(wrapper_func, 1)

# We get a sensible result!
Average_probability_of_lick = np.exp(res.x)[0]
sanity_check = len(licks)/(stop_time + 0.000001)

# evaluate fit by plotting prediction and lick times
def compare_model(res, time_vec, licks, licksdt, stop_time):
    fig,ax  = plt.subplots()    
    nll, latent = mean_lick_model(res.x, licksdt, stop_time)
    plt.plot(time_vec,latent,'b',label='model')
    plt.vlines(licks,0, 0.1, alpha = 0.3, label='licks')
    plt.ylim([0, .1])
    plt.xlim(600,660)
    plt.legend(loc=9 )
    plt.xlabel('time (s)')
    plt.ylabel('Licking Probability')
    plt.tight_layout()
    return fig, ax

# make a figure
compare_model(res, time_vec, licks, licksdt, stop_time)
