# First load the database of sessions, then find if we have models fit for each, then do 
# some extraction and save the results.
import os
import numpy as np
import pandas as pd
from glob import glob
from matplotlib import pyplot as plt
from licking_behavior.src import licking_model as mo
from allensdk.brain_observatory.behavior import behavior_ophys_session as bos
from allensdk.brain_observatory.behavior import behavior_session as bs
import importlib; importlib.reload(bs)
from allensdk.brain_observatory.behavior import stimulus_processing
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.internal.api import behavior_lims_api as bla

import sys; sys.path.append('/home/nick.ponvert/src/nick-allen')

# nick-utils
import mp
from colorpalette import get_colors
from extraplots import boxoff

vb_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/vb_sessions.h5', key='df')
storage_dir = '/home/nick.ponvert/nco_home/cluster_jobs/20190708_fit_training_history'

def find_model(row):
    if pd.isnull(row['ophys_experiment_id']):
        session_str = 'model_behavior_{}*'.format(int(row['behavior_session_id']))
    else:
        session_str = 'model_ophys_{}*'.format(int(row['ophys_experiment_id']))

    models = glob(os.path.join(storage_dir, session_str))
    return models

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

vb_sessions['model_fits'] = mp.parallelize_on_rows(vb_sessions, find_model)

has_model = list(map(len, vb_sessions['model_fits'].values))
vb_sessions['has_model'] = has_model

# Subject with 25 model fits
#  subject = 795512663
subject = 834823464
sessions_to_use = vb_sessions.query('donor_id == @subject and has_model==1')

##### FOR CALCULATING PERFORMANCE METRICS (BROKEN)
#  for ind_row, row in sessions_to_use.iterrows():
#      print('row {}'.format(ind_row))
#      session, session_id = load_session(row)
#      metrics = session.get_performance_metrics()
#      for key, val in metrics.items():
#          sessions_to_use.loc[ind_row, key] = val

# Want to see if the filter shapes are related to dprime. So, we will color by dprime value.

filters_to_plot = ['post_lick_mixed', 'reward', 'flash', 'change_flash']
plt.clf()
fig, axes = plt.subplots(4, 1)
plt.subplots_adjust(hspace=0.8)

colors = get_colors(len(sessions_to_use))
#  all_dprime = sessions_to_use['max_dprime'].values


for ind_iter, (ind_session, session) in enumerate(sessions_to_use.iterrows()):
    model_path = session['model_fits'][0]
    model = mo.unpickle_model(model_path)

    for filter_name in filters_to_plot:
        filter_obj = model.filters[filter_name]
        linear_filt = filter_obj.build_filter()
        nonlinear_filt = np.exp(linear_filt)
        filter_time_vec = filter_obj.filter_time_vec
        ax = axes[filters_to_plot.index(filter_name)]
        #  color_to_use = colors[np.argsort(all_dprime)[ind_iter]]
        color_to_use = colors[ind_iter]
        if filter_name=='post_lick_mixed':
            ax.plot(filter_time_vec, nonlinear_filt, color=color_to_use, label=session['stage_name'])
        else:
            ax.plot(filter_time_vec, nonlinear_filt, color=color_to_use)
        ax.set_title(filter_name)
        ax.set_ylabel('filter gain')
        ax.set_xlabel('time (s)')
        boxoff(ax)

    axes[0].legend(ncol=2, fontsize=8, frameon=False)

plt.show()

