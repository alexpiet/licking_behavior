#!/usr/bin/env python

# Import whatever you like
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import fit_tools
import pickle
import sys

if __name__ == '__main__': # Don't understand why you need this part
    name_of_this_file   = sys.argv[0]
    experiment_id       = sys.argv[1]
    
    ## Python Code Here
    dt = 0.01
    data = fit_tools.get_sdk_data(experiment_id, load_dir='/allen/aibs/technology/nicholasc/behavior_ophys')
    licks, licksdt, start_time, stop_time, time_vec, running_speed, rewardsdt, flashesdt, change_flashesdt = fit_tools.extract_sdk_data(data,dt)
   
    def wrapper_full(params):
        return fit_tools.licking_model(params, licksdt, stop_time, mean_lick_rate=True, dt = dt, 
post_lick=True,num_post_lick_params=10,post_lick_duration=.21, post_lick_sigma =0.025, 
include_running_speed=True, num_running_speed_params=6,running_speed=running_speed, 
include_reward=True, num_reward_params=40,reward_duration=4, reward_sigma = 0.1 ,rewardsdt=rewardsdt, 
include_flashes=True, num_flash_params=15,flash_duration=0.7, flash_sigma = 0.025, flashesdt=flashesdt, 
include_change_flashes=True, num_change_flash_params=15,change_flash_duration=0.7, change_flash_sigma = 0.025, change_flashesdt=[])
    
    def wrapper_func(params):
        return wrapper_full(params)[0]
    
    # Do Optimization
    inital_param = np.concatenate(([-.5],np.zeros((80,))))
    res = minimize(wrapper_func, inital_param)
    
    # Compute BIC, NLL, Latent
    res = fit_tools.evaluate_model(res,wrapper_full, licksdt, stop_time)
    
    # Save res
    filepath = "fitglm_"+str(experiment_id)  
    file_temp = open(filepath,'wb')
    pickle.dump(res, file_temp)
    file_temp.close()


