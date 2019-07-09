import sys
import numpy as np
import pandas as pd
from licking_behavior.src.generate_dataset import preprocess_data_sdk
from licking_behavior.src import licking_model as mo 
from licking_behavior.src import filters
from multiprocessing import  Pool
from functools import partial
from allensdk.brain_observatory.behavior import behavior_ophys_session as bos
from allensdk.brain_observatory.behavior import behavior_session as bs
from allensdk.brain_observatory.behavior import stimulus_processing
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.internal.api import behavior_lims_api as bla
import datetime

# I am going to pass in the row index (pandas index, not iterative index)
row_loc = int(sys.argv[1])

vb_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/vb_sessions.h5', key='df')

#Don't use sessions that we can't load (like the rf mapping ones)
sessions_to_use = vb_sessions[vb_sessions['stage_name'] != 'Load error']

output_dir = '/home/nick.ponvert/nco_home/cluster_jobs/20190708_fit_training_history'

def load_session(row):
    if pd.isnull(row['ophys_experiment_id']):
        api = bla.BehaviorLimsApi(int(row['behavior_session_id']))
        session_id = 'behavior_{}'.format(int(row['behavior_session_id']))
        session = bs.BehaviorSession(api)
    else:
        api = boa.BehaviorOphysLimsApi(int(row['ophys_experiment_id']))
        session_id = 'ophys_{}'.format(int(row['ophys_experiment_id']))
        session = bos.BehaviorOphysSession(api)
    return session, session_id

def fit_model(row):
    print("Starting session fit at {}".format(datetime.datetime.now()))
    session, session_id = load_session(row)

    (running_timestamps,
     running_speed,
     lick_timestamps,
     stim_on_timestamps,
     stim_off_timestamps,
     stim_id,
     stim_mapping,
     stim_omitted,
     reward_timestamps) = preprocess_data_sdk(session)

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
              verbose=False,
              name='{}'.format(session_id),
              l2=2)

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

    model.fit()
    model.dropout_analysis()
    model.save(output_dir)
    print("Done with fit")

row_to_fit = vb_sessions.loc[row_loc]
#  parallelize_on_rows(sessions_to_use, fit_model)
fit_model(row_to_fit)

