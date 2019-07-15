import os
import glob
from licking_behavior.src import licking_model as mo
import pandas as pd
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.internal.api import PostgresQueryMixin
import importlib; importlib.reload(boa)
importlib.reload(mo)

import numpy as np
from allensdk.brain_observatory.behavior import behavior_ophys_session as bos
from allensdk.internal.api import behavior_ophys_api as boa
from tqdm import tqdm

## Get experiment IDs to use
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
        os.date_of_acquisition as session_date,
        equipment.name as equipment_name

        FROM ophys_experiments_visual_behavior_experiment_containers oec
        LEFT JOIN ophys_experiments oe ON oe.id = oec.ophys_experiment_id
        LEFT JOIN ophys_sessions os ON oe.ophys_session_id = os.id
        LEFT JOIN specimens sp ON sp.id=os.specimen_id
        LEFT JOIN donors d ON d.id=sp.donor_id
        LEFT JOIN imaging_depths id ON id.id=os.imaging_depth_id
        LEFT JOIN structures st ON st.id=oe.targeted_structure_id
        LEFT JOIN equipment ON equipment.id=os.equipment_id
        '''

experiment_df = pd.read_sql(query, api.get_connection())

states_to_use = ['container_qc', 'passed']

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

# First pass
# subjects_to_use = [847076515]
                   
# Second pass
# subjects_to_use = [827778598, 814111925, 813703535, 820878203, 803258370]

# Third pass
#subjects_to_use = [800311507, 789992895, 772629800, 766025752, 766015379, 788982905,
# 843387577, 789014546, 830940312, 834902192, 820871399, 827778598]

# All subjects I've fit
subjects_to_use = [847076515, 827778598, 814111925, 813703535, 820878203, 803258370,
                   800311507, 789992895, 772629800, 766025752, 766015379, 788982905,
                   843387577, 789014546, 830940312, 834902192, 820871399, 827778598]

'''
           container_id  workflow_state    ...      model_fits  has_model
donor_id                                   ...
722884873             2               2    ...               2          2
756674776             1               1    ...               1          1
766015379             1               1    ...               1          1
772629800             3               3    ...               3          3
795512663            25              25    ...              25         25
803258370             9               9    ...               9          9
834823464            14              14    ...              14         14
842724844             7               7    ...               7          7
'''



conditions = [
    "workflow_state in @states_to_use",
    "equipment_name != 'MESO.1'",
    "donor_id in @subjects_to_use"
]

query_string = ' and '.join(conditions)
experiments_to_use = experiment_df.query(query_string)
##

## Append model_file to the experiments_to_use df
output_directory = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_jobs/20190629_sdk_fit/'

def get_pickle_path(experiment_id):
    path_list = glob.glob(os.path.join(output_directory, "model_{}*".format(experiment_id)))
    if len(path_list)>0:
        return path_list[0]
    elif len(path_list)==0:
        return None

def has_model(experiment_id):
    path_list = glob.glob(os.path.join(output_directory, "model_{}*".format(experiment_id)))
    if len(path_list)>0:
        return True
    elif len(path_list)==0:
        return False

experiments_to_use['model_file'] = experiments_to_use.apply(lambda row: get_pickle_path(row['ophys_experiment_id']), axis=1)
experiments_to_use['has_model'] = experiments_to_use.apply(lambda row: has_model(row['ophys_experiment_id']), axis=1)
##

model_fits = experiments_to_use[experiments_to_use['has_model']]

#  filter_metrics = []
#  for this_model in models_to_use:
#      model = mo.unpickle_model(this_model)
#      filter_metrics.append({this_model:model.filter_metrics()})

for ind_row, row in model_fits.iterrows():
    model = mo.unpickle_model(row['model_file'])
    
    metrics = model.filter_metrics()
    model_fits.loc[ind_row, 'lick_latency'] = metrics['post_lick']['time_to_peak']
    model_fits.loc[ind_row, 'lick_max'] = metrics['post_lick']['max_gain']
    model_fits.loc[ind_row, 'reward_latency'] = metrics['reward']['time_to_peak']
    model_fits.loc[ind_row, 'reward_max'] = metrics['reward']['max_gain']
    model_fits.loc[ind_row, 'flash_latency'] = metrics['flash']['time_to_peak']
    model_fits.loc[ind_row, 'flash_max'] = metrics['flash']['max_gain']
    model_fits.loc[ind_row, 'cf_latency'] = metrics['change_flash']['time_to_peak']
    model_fits.loc[ind_row, 'cf_max'] = metrics['change_flash']['max_gain']


# Add the performance metrics
cache_dir = '/home/nick.ponvert/nco_home/data/performance_metrics_cache'

for ind_row, row in model_fits.iterrows():
    experiment_id = row['ophys_experiment_id']
    fn = "{}_metrics.npz".format(experiment_id)
    full_path = os.path.join(cache_dir, fn)
    try:
        metrics_file = np.load(full_path)
    except FileNotFoundError:
        continue
    else:
        for key in metrics_file.files:
            model_fits.loc[ind_row, key] = metrics_file[key]


from matplotlib import pyplot as plt
import seaborn as sns
metrics = ['lick_latency', 'lick_max', 'reward_latency', 'reward_max',
  'flash_latency', 'flash_max', 'cf_latency', 'cf_max']

sns.set_palette("GnBu")
g = sns.PairGrid(model_fits, vars=metrics, hue='max_dprime')
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter);
plt.show()
