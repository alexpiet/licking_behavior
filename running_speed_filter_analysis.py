import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
plt.ion() # makes non-blocking figures
import fit_tools
import multiprocessing
import pickle

# Define which experiment id you want
experiment_id = 715887471

# Get the data
dt = 0.01 # 10msec timesteps
data = fit_tools.get_data(experiment_id, save_dir='./example_data')
licks = data['lick_timestamps']
running_timestamps = data['running_timestamps']
running_speed = data['running_speed']
rewards = np.round(data['reward_timestamps'],2)
rewardsdt = np.round(rewards*(1/dt))

# get start/stop time for session
start_time = 1
stop_time = int(np.round(running_timestamps[-1],2)*(1/dt))
licks = licks[licks < stop_time/100]
licks = np.round(licks,2)
licksdt = np.round(licks*(1/dt))
time_vec = np.arange(0,stop_time/100.0,dt)

# Let's look at 1-16 time bins first
n_bins = range(1, 17)

def model(n_bins):
    nll, latent = fit_tools.licking_model(np.concatenate(([-.5],np.zeros((n_bins,)))), licksdt, stop_time, post_lick=False,include_running_speed=True, num_running_speed_params=n_bins, running_speed=running_speed)

    # Wrapper function for optimization that only takes one input
    def running_wrapper_func(params): return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_speed=True, num_running_speed_params=n_bins,running_speed=running_speed)[0]

    # optimize
    inital_param = np.concatenate(([-.5],np.zeros((n_bins,))))
    res_running = minimize(running_wrapper_func, inital_param)

    def wrapper(params):
        return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_speed=True, num_running_speed_params=n_bins,running_speed=running_speed)

    # Evaluate the model
    res_running = fit_tools.evaluate_model(res_running,wrapper, licksdt, stop_time)
    # fit_tools.compare_model(res_running.latent, time_vec, licks, stop_time, running_speed)

    #Save the output
    fn = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/padded_run_model_{}.pkl'.format(n_bins)
    print('pickling {}'.format(fn))
    with open('file.txt', 'wb') as pfile:
        pickle.dump(res_running, pfile)
    return res_running.BIC

pool = multiprocessing.Pool(16) # Have 16 cores
BICs = pool.map(model, n_bins)

#fit_tools.build_filter(, np.arange(dt,.20,dt), 0.025, plot_filters=True,plot_nonlinear=True)

#plt.plot(res_running.x[1:21], 'k-')
