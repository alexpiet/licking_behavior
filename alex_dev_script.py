import pickle
from alex_utils import load
import matplotlib.pyplot as plt
import fit_tools
plt.ion()
import numpy as np
import matplotlib
from scipy.optimize import minimize

filepath = '/Users/alex.piet/glm_fits/'
experiment_ids = [837729902, 838849930,836910438,840705705,840157581,841601446,840702910,841948542,841951447,842513687,842973730,843519218,846490568,847125577,848697604] 

ids = experiment_ids[-1]
res = load(filepath+'fitglm_'+str(ids))
dt = 0.01
data = fit_tools.get_data(ids, save_dir='/allen/programs/braintv/workgroups/nc-ophys/alex.piet')
licks, licksdt, start_time, stop_time, time_vec, running_speed, rewardsdt, flashesdt, change_flashesdt,running_acceleration = fit_tools.extract_data(data,dt)
flashes = flashesdt/100
rewards = rewardsdt/100
change_flashes = change_flashesdt/100


fit_tools.compare_model(res.latent[:-1], time_vec, licks, stop_time, rewards=rewards, flashes=flashes, change_flashes=change_flashes, running_speed = running_speed[:-1], running_acceleration=running_acceleration[:-1])

reload(plot_tools)
plt.close('all')
params = [res.x[57:72],res.x[57:72],res.x[17:57],res.x[1:11]]
durations   = [.76,1.6,4,.21]
sigmas      = [0.05,0.05,.25,0.025]
events      = [[0, 75,150,225,300,375,450,525],[100],[134],[134,150,162,175,189,201,214,229]]
plot_tools.plot_components(params,durations,sigmas,events,5)



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


