import os
import numpy as np
from matplotlib import pyplot as plt

experiment_id = 715887471

save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor'
output_fn = 'experiment_{}.npz'.format(experiment_id)
full_path = os.path.join(save_dir, output_fn)

data = np.load(full_path)

running_timestamps = data['running_timestamps']
running_speed = data['running_speed']
lick_timestamps = data['lick_timestamps']
stim_on_timestamps = data['stim_on_timestamps']
stim_off_timestamps = data['stim_off_timestamps']
stim_name = data['stim_name']
stim_omitted = data['stim_omitted']
reward_timestamps = data['reward_timestamps']

