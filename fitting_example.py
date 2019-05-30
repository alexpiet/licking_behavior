import matplotlib.pyplot as plt
import numpy as np

plt.ion() # makes non-blocking figures
from alex_utils import whos

# Import relevant functions
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

# Define which experiment id you want
ophys_experiment_id = 783927872

# Option 2 - Pipeline NWB file:
api = BehaviorOphysLimsApi(ophys_experiment_id)
filepath = api.get_nwb_filepath()
nwb_api = BehaviorOphysNwbApi(filepath)
session = BehaviorOphysSession(api)
print(session.metadata)

# plot running speed
plt.plot(session.running_speed.timestamps, session.running_speed.values)





