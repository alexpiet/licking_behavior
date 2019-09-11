# First load the database of sessions, then find if we have models fit for each, then do 
# some extraction and save the results.
import os
import numpy as np
import pandas as pd
from allensdk.brain_observatory.behavior import behavior_ophys_session as bos
from allensdk.brain_observatory.behavior import behavior_session as bs
import importlib; importlib.reload(bs)
from allensdk.brain_observatory.behavior import stimulus_processing
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.internal.api import behavior_lims_api as bla
from tqdm import tqdm

import sys; sys.path.append('/home/nick.ponvert/src/nick-allen')

# nick-utils
import mp
from colorpalette import get_colors
from extraplots import boxoff

vb_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/vb_sessions.h5', key='df')

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

case = 0
if case==0: 
    for ind_row, row in vb_sessions.iloc[:1].iterrows():
        session, session_id = load_session(row)

elif case==1:

    failures = []
    for ind_row, row in tqdm(vb_sessions.iterrows(), total=vb_sessions.shape[0]):
        session, session_id = load_session(row)
        try:
            metrics = session.get_performance_metrics()
            for key, val in metrics.items():
                sessions_to_use.loc[ind_row, key] = val
        except:
            failures.append(ind_row)
            continue

