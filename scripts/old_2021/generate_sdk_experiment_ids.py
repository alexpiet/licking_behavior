import numpy as np
import os
import pandas as pd
from allensdk.internal.api import PostgresQueryMixin

## Get the visual behavior experiment container sessions
## Have to limit to just 'VisualBehavior' project code to exclude 
## some visual coding targeted experiments that are in there
api = PostgresQueryMixin()
query = '''
        SELECT

        oec.visual_behavior_experiment_container_id as container_id,
        oec.ophys_experiment_id,
        oe.workflow_state,
        d.full_genotype as full_genotype,
        d.id as donor_id,
        id.depth as imaging_depth,
        st.acronym as targeted_structure,
        os.name as session_name,
        equipment.name as equipment_name

        FROM ophys_experiments_visual_behavior_experiment_containers oec
        LEFT JOIN ophys_experiments oe ON oe.id = oec.ophys_experiment_id
        LEFT JOIN ophys_sessions os ON oe.ophys_session_id = os.id
        LEFT JOIN specimens sp ON sp.id=os.specimen_id
        LEFT JOIN donors d ON d.id=sp.donor_id
        LEFT JOIN imaging_depths id ON id.id=os.imaging_depth_id
        LEFT JOIN structures st ON st.id=oe.targeted_structure_id
        LEFT JOIN equipment ON equipment.id=os.equipment_id
        LEFT JOIN projects p ON p.id=os.project_id

        WHERE p.code = 'VisualBehavior'
        '''

## Additionally limit by qc state, and drop mesoscope sessions because we 
## can't load the data with the SDK
experiment_df = pd.read_sql(query, api.get_connection())
states_to_use = ['container_qc', 'passed']

conditions = [
    "workflow_state in @states_to_use",
    "equipment_name != 'MESO.1'",
]

query_string = ' and '.join(conditions)
experiments_to_use = experiment_df.query(query_string).copy()

## Iterate through the dataframe of experiments to use and get 
## performance metrics.

cache_dir = '/home/nick.ponvert/nco_home/data/performance_metrics_cache'

for ind_row, row in experiments_to_use.iterrows():
    experiment_id = row['ophys_experiment_id']
    fn = "{}_metrics.npz".format(experiment_id)
    full_path = os.path.join(cache_dir, fn)
    try:
        metrics_file = np.load(full_path)
    except FileNotFoundError:
        continue
    else:
        for key in metrics_file.files:
            experiments_to_use.loc[ind_row, key] = metrics_file[key]

#  metrics = ['trial_count',
#   'go_trial_count',
#   'catch_trial_count',
#   'hit_trial_count',
#   'miss_trial_count',
#   'false_alarm_trial_count',
#   'correct_reject_trial_count',
#   'auto_rewarded_trial_count',
#   'total_reward_count',
#   'total_reward_volume',
#   'maximum_reward_rate',
#   'engaged_trial_count',
#   'mean_hit_rate',
#   'mean_hit_rate_engaged',
#   'mean_false_alarm_rate',
#   'mean_false_alarm_rate_engaged',
#   'mean_dprime',
#   'mean_dprime_engaged',
#   'max_dprime',
#   'max_dprime_engaged']

metrics = [
 'hit_trial_count',
 'miss_trial_count',
 'false_alarm_trial_count',
 'correct_reject_trial_count',
 'total_reward_volume',
 'maximum_reward_rate',
 'engaged_trial_count',
 'mean_dprime',
 'mean_dprime_engaged',
 'max_dprime',
 'max_dprime_engaged']

from matplotlib import pyplot as plt
import seaborn as sns
g = sns.PairGrid(experiments_to_use, vars=metrics)
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter, s=3);
g.fig.set_size_inches(20,20)
g.savefig('/home/nick.ponvert/pairplot_test.png')




#  for ind_row, row in experiments_to_use.iterrows():
#      get_session_performance_metrics(row)
    

#  for ind_row, row in experiments_to_use.iterrows():
#      api = boa.BehaviorOphysLimsApi(int(row['ophys_experiment_id']))
#      session = bos.BehaviorOphysSession(api)
#      try:
#          metrics = session.get_performance_metrics()
#          for key, val in metrics.items():
#              experiments_to_use.loc[ind_row, key] = val
#      # KeyErrors for RF mapping sessions, no data['items']['behavior']
#      # OSErrors because some files are truncated?
#      except (KeyError, OSError):
#          "error with row {}"

# Unique donor ids
#  [800311507, 789992895, 772629800, 766025752, 766015379, 788982905,
#   843387577, 789014546, 830940312, 834902192, 820871399, 827778598,
#   830901414, 842724844, 765991998, 803592122, 791756316, 795522266,
#   791871796, 847074139, 800312319, 837628429, 813702144, 795512663,
#   772622642, 834823464, 784057617, 837581576, 807248984, 847076515,
#   814111925, 813703535, 830896318, 831004160, 820878203, 803258370,
#   823826963, 760949537, 803582244, 722884873, 744935640, 738695281,
#   756575004, 831008730, 741953980, 756577240, 744911447, 756674776,
#   722884863, 623289718, 652074239, 651725149, 713968352, 710324779,
#   713426152, 707282079, 710269817, 716520602, 719239251, 734705385,
#   705093661, 692785809]

#  conditions = [
#      "workflow_state in @states_to_use",
#      "equipment_name != 'MESO.1'",
#      "donor_id in @subjects_to_use"
#  ]

# First pass
# subjects_to_use = [847076515]
                   
# Second pas
# subjects_to_use = [827778598, 814111925, 813703535, 820878203, 803258370]

# Third pass
#  subjects_to_use = [800311507, 789992895, 772629800, 766025752, 766015379, 788982905,
#   843387577, 789014546, 830940312, 834902192, 820871399, 827778598]

# Fourth pass, doing all the rest. This is a lot of subjects.
#  subjects_to_use = [
#  830901414, 842724844, 765991998, 803592122, 791756316, 795522266,
#  791871796, 847074139, 800312319, 837628429, 813702144, 795512663,
#  772622642, 834823464, 784057617, 837581576, 807248984, 847076515,
#  814111925, 813703535, 830896318, 831004160, 820878203, 803258370,
#  823826963, 760949537, 803582244, 722884873, 744935640, 738695281,
#  756575004, 831008730, 741953980, 756577240, 744911447, 756674776,
#  722884863, 623289718, 652074239, 651725149, 713968352, 710324779,
#  713426152, 707282079, 710269817, 716520602, 719239251, 734705385,
#  705093661, 692785809]
#  
#  conditions = [
#      "workflow_state in @states_to_use",
#      "equipment_name != 'MESO.1'",
#      "donor_id in @subjects_to_use"
#  ]
#  
#  query_string = ' and '.join(conditions)
#  experiments_to_use = experiment_df.query(query_string)
#  
#  experiment_ids = experiments_to_use['ophys_experiment_id'].unique() #Remove any duplicates
#  print(experiment_ids)