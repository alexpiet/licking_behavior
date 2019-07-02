import numpy as np
import sys
from licking_behavior.src import generate_dataset
from licking_behavior.src import licking_model as mo
from licking_behavior.src import filters

#  experiment_id = 841951447
experiment_id = sys.argv[1]
output_dir = '/home/nick.ponvert/nco_home/cluster_jobs/20190629_sdk_fit'

# Pull the data we need for the session using the sdk LIMS api
(running_timestamps,
 running_speed,
 lick_timestamps,
 stim_on_timestamps,
 stim_off_timestamps,
 stim_id,
 stim_mapping,
 stim_omitted,
 reward_timestamps) = generate_dataset.preprocess_data_sdk(experiment_id)

# Push the data into our format
data = {}
data['lick_timestamps'] = lick_timestamps 
data['running_timestamps'] = running_timestamps
data['running_speed'] = running_speed
data['stim_id'] = stim_id
data['reward_timestamps'] = reward_timestamps
data['stim_on_timestamps'] = stim_on_timestamps

# use binning function in licking model to further preprocess
# TODO: We should make this func not take the data object. Let's just pass arrs
(licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
 running_speed, running_timestamps, running_acceleration,
 timebase, time_start, time_end) = mo.bin_data(data, dt=0.01)


model = mo.Model(dt=0.01,
              licks=licks_vec, 
              verbose=True,
              name='{}'.format(experiment_id),
              l2=0.5)

#  post_lick_filter = mo.GaussianBasisFilter(data = licks_vec,
#                                         dt = model.dt,
#                                         **filters.post_lick)
#  model.add_filter('post_lick', post_lick_filter)

long_lick_filter = mo.GaussianBasisFilter(data = licks_vec,
                                       dt = model.dt,
                                       **filters.long_lick)
model.add_filter('post_lick', long_lick_filter)

reward_filter = mo.GaussianBasisFilter(data = rewards_vec,
                                    dt = model.dt,
                                    **filters.reward)
model.add_filter('reward', reward_filter)

#  long_reward_filter = mo.GaussianBasisFilter(data = rewards_vec,
#                                      dt = model.dt,
#                                      **filters.long_reward)
#  model.add_filter('long_reward', long_reward_filter)

flash_filter = mo.GaussianBasisFilter(data = flashes_vec,
                                   dt = model.dt,
                                   **filters.flash)
model.add_filter('flash', flash_filter)

change_filter = mo.GaussianBasisFilter(data = change_flashes_vec,
                                    dt = model.dt,
                                    **filters.change)
model.add_filter('change_flash', change_filter)

model.fit()
model.save(output_dir)
