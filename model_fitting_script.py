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

    def wrapper_full(params):
        return licking_model(params, licksdt, stop_time, mean_lick_rate=True, dt = 0.01,
    post_lick=True,num_post_lick_params=10,post_lick_duration=.21, post_lick_sigma =0.025, 
    include_running_speed=True, num_running_speed_params=10,running_speed_duration = 0.25, running_speed_sigma = 0.025,running_speed=0,
    include_reward=True, num_reward_params=10,reward_duration =4, reward_sigma = 0.5 ,rewardsdt=[],
    include_flashes=True, num_flash_params=10,flash_duration=0.750, flash_sigma = 0.025, flashesdt=[],
    include_change_flashes=True, num_change_flash_params=10,change_flash_duration=0.750, change_flash_sigma = 0.025, change_flashesdt=[])

    def wrapper_func(params):
        return wrapper_full(params)[0]

    # Do Optimization
    inital_param = np.concatenate(([-.5],np.ones((50,))))
    res = minimize(wrapper_func, inital_param)

    # Compute BIC, NLL, Latent
    res = fit_tools.evaluate_model(res,wrapper_full, licksdt, stop_time)

    # Save res
    ........
