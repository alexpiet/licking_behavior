#!/usr/bin/env python

# Import whatever you like
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import fit_tools

if __name__ == '__main__': # Don't understand why you need this part
    name_of_this_file   = sys.argv[0]
    experiment_i        = sys.argv[1]

    ## Python Code Here
    data = fit_tools.get_data(experiment_id, save_dir='./example_data')
    licks = data['lick_timestamps']
    running_timestamps = data['running_timestamps']
    running_speed = data['running_speed']
    rewards = np.round(data['reward_timestamps'],2)
    dt = 0.01

    # get start/stop time for session
    start_time = 1
    stop_time = int(np.round(running_timestamps[-1],2)*(1/dt))
    licks = licks[licks < stop_time/100]
    licks = np.round(licks,2)
    licksdt = np.round(licks*(1/dt))
    rewardsdt = np.round(rewards*(1/dt))
    time_vec = np.arange(0,stop_time/100.0,dt)

    def wrapper_func(params):
        return fit_tools.licking_model(params, licksdt, stop_time, post_lick=True,
include_running_speed=True, 
include_reward=True, num_reward_params=20, reward_duration=4, reward_sigma=0.5,rewardsdt=rewardsdt)[0]

    # Do Optimization
    inital_param = np.concatenate(([-.5],np.ones((20,))))
    res = minimize(wrapper_func, inital_param)

    def wrapper_full(params):
        return fit_tools.licking_model(params, licksdt, stop_time, post_lick=False,include_running_speed=False, include_reward=True, num_reward_params=20, reward_duration=4, reward_sigma=0.5,rewardsdt=rewardsdt)

    # Compute BIC, NLL, Latent
    res = fit_tools.evaluate_model(res,wrapper_full, licksdt, stop_time)

    # Save res

