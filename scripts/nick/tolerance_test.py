import numpy as np
from licking_behavior.src import licking_model as mo
import importlib; importlib.reload(mo)
from licking_behavior.src import fit_tools
from licking_behavior.src import filters
import importlib; importlib.reload(filters)
importlib.reload(mo)
import cProfile

dt = 0.01

experiment_id = 715887471
data = fit_tools.get_data(experiment_id, save_dir='../../example_data')

(licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
 running_speed, running_timestamps, running_acceleration, timebase,
 time_start, time_end) = mo.bin_data(data, dt, time_start=300, time_end=1000)


model = mo.Model(dt=0.01,
              licks=licks_vec, 
              verbose=True,
              name='{}'.format(experiment_id),
              l2=0.1)
long_lick_filter = mo.MixedGaussianBasisFilter(data = licks_vec,
                                            dt = model.dt,
                                            **filters.long_lick_mixed)
model.add_filter('post_lick_mixed', long_lick_filter)

reward_filter = mo.GaussianBasisFilter(data = rewards_vec,
                                    dt = model.dt,
                                    **filters.long_reward)
model.add_filter('reward', reward_filter)

flash_filter = mo.GaussianBasisFilter(data = flashes_vec,
                                   dt = model.dt,
                                   **filters.flash)
model.add_filter('flash', flash_filter)

change_filter = mo.GaussianBasisFilter(data = change_flashes_vec,
                                    dt = model.dt,
                                    **filters.change)
model.add_filter('change_flash', change_filter)

model.initialize_filters() # zero out all filter params
cProfile.run('model.fit()')
# model.fit()
