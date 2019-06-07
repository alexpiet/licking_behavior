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

nll, latent = fit_tools.licking_model([-.5,0,0,0,0,0], licksdt, stop_time, post_lick=False, include_running_acceleration=True, running_acceleration=running_acceleration)

# Wrapper function for optimization that only takes one input
def wrapper_func(params):
    #return mean_lick_model(mean_lick_rate,licksdt,stop_time)[0]
    return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_acceleration=True, running_acceleration=running_acceleration)[0]

# optimize
inital_param = np.zeros((6,))
res_mean = minimize(wrapper_func, inital_param)

x= fit_tools.build_filter(res_mean.x[1:], np.arange(dt,1.01,dt), 0.25, plot_filters=True)


