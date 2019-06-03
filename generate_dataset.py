import os
import numpy as np
from matplotlib import pyplot as plt

from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset

experiment_id = 715887471
#ophys_data, dataset, analysis,stims,fr = get_session(experiment_id)
cache_dir = r'/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/visual_behavior_production_analysis'

# Convert the session if necessary, and build the dataset object
# ophys_data = convert_level_1_to_level_2(experiment_id, cache_dir, plot_roi_validation=False)
dataset = VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)

# Running timestamps and speed values
running_timestamps = dataset.running_speed['time']
running_speed = dataset.running_speed['running_speed']

# Lick timestamps
lick_timestamps = dataset.licks['time']

# Timestamps of stim on, off, id, and whether or not the stim was omitted
stim_on_timestamps = dataset.stimulus_table['start_time']
stim_off_timestamps = dataset.stimulus_table['end_time']
stim_name = dataset.stimulus_table['image_name']
stim_omitted = dataset.stimulus_table['omitted']

# Reward delivery timestamps
reward_timestamps = dataset.rewards['time']

# Save data arrays out to NPZ format
save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor'
output_fn = 'experiment_{}.npz'.format(experiment_id)
full_path = os.path.join(save_dir, output_fn)
np.savez(full_path, 
        running_timestamps = running_timestamps,
        running_speed = running_speed,
        lick_timestamps = lick_timestamps,
        stim_on_timestamps = stim_on_timestamps,
        stim_off_timestamps = stim_off_timestamps,
        stim_name = stim_name,
        stim_omitted = stim_omitted,
        reward_timestamps = reward_timestamps)


