import os
import pickle
os.chdir('/home/alex.piet/codebase/behavior/licking_behavior')
os.chdir('/Users/alex.piet/licking_behavior')
from alex_utils import load
import matplotlib.pyplot as plt
import fit_tools
plt.ion()
import numpy as np
import matplotlib
from scipy.optimize import minimize
import plot_tools
from importlib import reload

filepath = '/Users/alex.piet/glm_fits/'
experiment_ids = [837729902, 838849930,836910438,840705705,840157581,841601446,840702910,841948542,841951447,842513687,842973730,843519218,846490568,847125577,848697604] 

ids = experiment_ids[-1]
ids = 841601446
# Load old data
#res = load(filepath+'fitglm_'+str(ids))
# Load model Object
fit_path = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/cluster_jobs'
Fn = 'glm_model_vba_v2_'+str(ids)+'.pkl'
full_path = os.path.join(fit_path, Fn)
print(str(ids))
model = fit_tools.Model.from_file_rebuild(full_path)
#model.plot_all_filters()



# Plot schematic figure
post_lick_params= model.res.x[1:11]
reward_params   = model.res.x[17:37]
flash_params    = model.res.x[37:52]
cflash_params   = model.res.x[52:67]
params      = [flash_params, cflash_params,reward_params,post_lick_params]
durations   = [model.flash_duration,model.change_flash_duration,model.reward_duration,model.post_lick_duration]
sigmas      = [model.flash_sigma,model.change_flash_sigma,model.reward_sigma,model.post_lick_sigma]
events      = [[0, 75,150,225,300,375,450,525],[75],[109],[109, 125,141,157,172, 190,205,230]]
plt.close('all')
reload(plot_tools)
plot_tools.plot_components(params,durations,sigmas,events,5,model.res.x[0])
plt.savefig('test.svg')



# Get Data, and plot session
dt = 0.01
data = fit_tools.get_data(ids, save_dir='/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor')
licks, licksdt, start_time, stop_time, time_vec, running_speed, rewardsdt, flashesdt, change_flashesdt,running_acceleration = fit_tools.extract_data(data,dt)
flashes = flashesdt/100
rewards = rewardsdt/100
change_flashes = change_flashesdt/100

reload(fit_tools)
plt.close('all')
fit_tools.compare_model(model.res.latent, time_vec, licks, stop_time, rewards=rewards, flashes=flashes, change_flashes=change_flashes, running_speed = running_speed[:-1])

# Sanity Checks
import analysis_tools as at
at.compare_all_inter_licks()
at.compare_dist(variable='licks')
at.compare_dist(variable='rewards')
at.compare_dist(variable='flash')
at.compare_dist(variable='change_flash')
at.compare_dist(variable='running_speed')
at.compare_dist(variable='running_acceleration')


# Optimization scraps
######
nll, latent = fit_tools.licking_model([-.5,0,0,0,0,0], licksdt, stop_time, post_lick=False, include_running_acceleration=True, running_acceleration=running_acceleration)

# Wrapper function for optimization that only takes one input
def wrapper_func(params):
    #return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_acceleration=True, running_acceleration=running_acceleration)[0]

# optimize
inital_param = np.zeros((6,))
res_mean = minimize(wrapper_func, inital_param)

x= fit_tools.build_filter(res_mean.x[1:], np.arange(dt,1.01,dt), 0.25, plot_filters=True)


