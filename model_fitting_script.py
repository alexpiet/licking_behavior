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

    model = fit_tools.Model(licks=data['licks'],
                            running_timestamps=data['running_timestamps'],
                            running_speed=data['running_speed'],
                            rewards=data['reward_timestamps'],
                            flashes=data['stim_flash_start'],
                            change_flashes=data["stim_change_time"],
                            include_post_lick=True,
                            include_running_speed=True,
                            include_flashes=True,
                            include_change_flashes=True)

    model.fit()
    model.save('glm_fit_{}.pkl'.format(experiment_id))


    #      def wrapper_full(params):
    #          return fit_tools.licking_model(params, licksdt, stop_time, mean_lick_rate=True, dt = dt, 
    #  post_lick=True,num_post_lick_params=10,post_lick_duration=.21, post_lick_sigma =0.025, 
    #  include_running_speed=True, num_running_speed_params=6,running_speed=running_speed, 
    #  include_reward=True, num_reward_params=40,reward_duration=4, reward_sigma = 0.1 ,rewardsdt=rewardsdt, 
    #  include_flashes=True, num_flash_params=15,flash_duration=0.76, flash_sigma = 0.05, flashesdt=flashesdt, 
    #  include_change_flashes=True, num_change_flash_params=15,change_flash_duration=0.76, change_flash_sigma = 0.05, change_flashesdt=change_flashesdt)
    
    
    # Save res
    #  filepath = "fitglm_"+str(experiment_id)  
    #  file_temp = open(filepath,'wb')
    #  pickle.dump(res, file_temp)
    #  file_temp.close()


