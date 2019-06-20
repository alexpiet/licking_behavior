import fit_tools
import numpy as np
from importlib import reload
reload(fit_tools)

experiment_id = 715887471

# Get the data
dt = 0.01 # 10msec timesteps
data = fit_tools.get_data(experiment_id, save_dir='./example_data')
licks = data['lick_timestamps']
running_timestamps = data['running_timestamps']
running_speed = data['running_speed']
rewards = np.round(data['reward_timestamps'],2)
flashes=np.round(data['stim_on_timestamps'],2)
dt = 0.01
rewardsdt = np.round(rewards*(1/dt))
flashesdt = np.round(flashes*(1/dt))

stims = data['stim_id']
stims[np.array(stims) == 8 ] = 100
diffs = np.diff(stims)
diffs[(diffs > 50) | (diffs < -50 )] = 0
diffs[ np.abs(diffs) > 0] = 1
diffs = np.concatenate([[0], diffs])

change_flashes = flashes[diffs == 1]
change_flashesdt = np.round(change_flashes*(1/dt))


# get start/stop time for session
start_time = 1
stop_time = int(np.round(running_timestamps[-1],2)*(1/dt))
licks = licks[licks < stop_time/100]
licks = np.round(licks,2)
licksdt = np.round(licks*(1/dt))
time_vec = np.arange(0,stop_time/100.0,dt)

model = fit_tools.Model(licks, running_timestamps, running_speed,
                        post_lick=True, include_running_speed=False,
                        include_reward=False, include_flashes=False,
                        include_change_flashes=False)
#model.fit()

