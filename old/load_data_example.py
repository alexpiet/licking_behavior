
# Import relevant functions
from allensdk.brain_observatory.behavior.behavior_ophys_session import BehaviorOphysSession
from allensdk.brain_observatory.behavior.behavior_ophys_api.behavior_ophys_nwb_api import BehaviorOphysNwbApi
from allensdk.internal.api.behavior_ophys_api import BehaviorOphysLimsApi

# Define which experiment id you want
ophys_experiment_id = 783927872

# Option 1 - Load data from LIMS direction:
api = BehaviorOphysLimsApi(ophys_experiment_id)
session = BehaviorOphysSession(api)
print(session.metadata)

# Option 2 - Pipeline NWB file:
api = BehaviorOphysLimsApi(ophys_experiment_id)
filepath = api.get_nwb_filepath()
nwb_api = BehaviorOphysNwbApi(filepath)
session = BehaviorOphysSession(api)
print(session.metadata)

# Option 3 - Off-pipeline NWB file:
filepath = '/allen/aibs/technology/nicholasc/behavior_ophys/behavior_ophys_session_{}.nwb'.format(ophys_experiment_id)
nwb_api = BehaviorOphysNwbApi(filepath)
session = BehaviorOphysSession(api)
print(session.metadata)






