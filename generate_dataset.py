import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

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
lick_timestamps = lick_timestamps[lick_timestamps>min(running_timestamps)]

# Timestamps of stim on, off, id, and whether or not the stim was omitted
stim_on_timestamps = dataset.stimulus_table['start_time']

#Only include stims that were presented after we started recording the running.
stims_to_include = stim_on_timestamps>min(running_timestamps)
stim_on_timestamps = stim_on_timestamps[stims_to_include]
stim_off_timestamps = dataset.stimulus_table['end_time']
stim_off_timestamps = stim_off_timestamps[stims_to_include]
stim_name = dataset.stimulus_table['image_name']
stim_name = stim_name[stims_to_include]
stim_omitted = dataset.stimulus_table['omitted']
stim_omitted = stim_omitted[stims_to_include]

# Reward delivery timestamps
reward_timestamps = dataset.rewards['time']
reward_timestamps = reward_timestamps[reward_timestamps>min(running_timestamps)]

# Subtract the start of running from all the arrays we save out. 
start_time = running_timestamps[0]

running_timestamps = running_timestamps - start_time
lick_timestamps = lick_timestamps - start_time
stim_on_timestamps = stim_on_timestamps - start_time
stim_off_timestamps = stim_off_timestamps - start_time
reward_timestamps = reward_timestamps - start_time

# For continuous measurements, interpolate to 10ms timebins
dt = 0.01
timebase_interpolation = np.arange(0, max(running_timestamps), dt)
f_running = interp1d(running_timestamps, running_speed)
running_speed_interpolated = f_running(timebase_interpolation)

# Save data arrays out to NPZ format
save_dir = r'/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/thursday_harbor'
output_fn = 'experiment_{}.npz'.format(experiment_id)
full_path = os.path.join(save_dir, output_fn)
np.savez(full_path, 
        running_timestamps = timebase_interpolation,
        running_speed = running_speed_interpolated,
        lick_timestamps = lick_timestamps,
        stim_on_timestamps = stim_on_timestamps,
        stim_off_timestamps = stim_off_timestamps,
        stim_name = stim_name,
        stim_omitted = stim_omitted,
        reward_timestamps = reward_timestamps)


