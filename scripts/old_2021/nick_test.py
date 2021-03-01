import numpy as np
import fit_tools

data = fit_tools.get_data(experiment_id, save_dir='./example_data')
licks = data['lick_timestamps']
running_timestamps = data['running_timestamps']
running_speed = data['running_speed']

# get start/stop time for session
start_time = 1
dt = 0.01 # 10msec timesteps
stop_time = int(np.round(running_timestamps[-1],2)*(1/dt))
licks = licks[licks < stop_time/100]

# mean_lick rate: scalar parameter that is the log(average-lick rate)
# mean_lick rate: scalar parameter that is the log(average-lick rate)
# licksdt: a vector of lick times in dt-index points
# stop_time: The index of the last time-bin
#
# Model with Mean lick rate, and post-lick filter
# params[0]: mean lick rate
# params[1:]: post-lick filter

# Returns: the NLL of the model, and the latent rate
def mean_running_model(params, run_speed, stop_time):
    '''
    Args:
        params: Model parameters.
                params[0]: Mean lick rate
                params[1:]: n-back running speed filter
        run_speed: vector of running speed for each time bin
        stop_time: index of the last time bin (should be n_bins?)
    '''

    # Mean lick rate base
    mean_lick_rate = params[0]
    base = np.ones(stop_time) * mean_lick_rate
