#!/usr/bin/env python

# Import whatever you like
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import fit_tools
import pickle
import sys
import os

import filters
import importlib; importlib.reload(filters)
import new_model_obj as mo

if __name__ == '__main__':
    name_of_this_file   = sys.argv[0]
    experiment_id       = sys.argv[1]
    
    ## Python Code Here
    dt = 0.01

    data = fit_tools.get_data(experiment_id)

    (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
     running_speed, running_timestamps, running_acceleration, timebase,
     time_start, time_end) = mo.bin_data(data, dt)

    param_save_fn = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/model_test_71_saved_params.npz' 

    output_dir = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_jobs/20190614_glm_fit'
    model = mo.Model(dt=0.01,
                  licks=licks_vec, 
                  verbose=True,
                  name='{}'.format(experiment_id),
                  l2=0.5)

    post_lick_filter = mo.GaussianBasisFilter(data = licks_vec,
                                           dt = model.dt,
                                           **filters.post_lick)
    model.add_filter('post_lick', post_lick_filter)

    reward_filter = mo.GaussianBasisFilter(data = rewards_vec,
                                        dt = model.dt,
                                        **filters.reward)
    model.add_filter('reward', reward_filter)

    flash_filter = mo.GaussianBasisFilter(data = flashes_vec,
                                       dt = model.dt,
                                       **filters.flash)
    model.add_filter('flash', flash_filter)

    change_filter = mo.GaussianBasisFilter(data = change_flashes_vec,
                                        dt = model.dt,
                                        **filters.change)
    model.add_filter('change_flash', change_filter)

    running_speed_filter= mo.GaussianBasisFilter(data = running_speed,
                                              dt = model.dt,
                                              **filters.running_speed)
    model.add_filter('running_speed', running_speed_filter)

    acceleration_filter= mo.GaussianBasisFilter(data = running_acceleration,
                                             dt = model.dt,
                                             **filters.acceleration)
    model.add_filter('acceleration', acceleration_filter)
    model.set_filter_params_from_file(param_save_fn)
    model.fit()
    model.save_filter_params(output_dir)