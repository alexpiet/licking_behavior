import pandas as pd
# I am going to go straight to LIMS to ask for all the behavior sessions associated with visual behavior containers. 
from allensdk.internal.api import PostgresQueryMixin

def get_container_behavior_sessions():

    api = PostgresQueryMixin()
    query = '''
    SELECT
    vbc.id AS container_id,
    vbc.workflow_state,
    d.id AS donor_id,
    bs.id AS behavior_session_id,
    bs.created_at AS date,
    bs.ophys_session_id,
    oe.id as ophys_experiment_id,
    os.name,
    p.code as project_code

    FROM visual_behavior_experiment_containers vbc
    LEFT JOIN specimens sp on sp.id = vbc.specimen_id
    LEFT JOIN donors d on d.id = sp.donor_id
    LEFT JOIN behavior_sessions bs ON bs.donor_id = d.id
    LEFT JOIN ophys_sessions os on os.id = bs.ophys_session_id
    LEFT JOIN ophys_experiments oe on oe.ophys_session_id = os.id
    LEFT JOIN projects p ON p.id=os.project_id
    '''

    return pd.read_sql(query, api.get_connection())

def get_visual_behavior_donors():

    api = PostgresQueryMixin()
    query = '''
    SELECT DISTINCT
    d.id AS donor_id

    FROM visual_behavior_experiment_containers vbc
    LEFT JOIN specimens sp on sp.id = vbc.specimen_id
    LEFT JOIN donors d on d.id = sp.donor_id
    LEFT JOIN behavior_sessions bs ON bs.donor_id = d.id
    LEFT JOIN ophys_sessions os on os.id = bs.ophys_session_id
    LEFT JOIN projects p ON p.id=os.project_id
    WHERE p.code = 'VisualBehavior'
    '''
    return pd.read_sql(query, api.get_connection())

# TODO: check out https://pandas.pydata.org/pandas-docs/stable/user_guide/integer_na.html to keep dtypes int even with nans
behavior_session_df = get_container_behavior_sessions()
behavior_session_df['workflow_state'].unique() #The different states for containers
not_failed = behavior_session_df.query("workflow_state != 'failed'") # Let's not work with failed containers

all_donors = not_failed['donor_id'].unique() #The different animal donor IDs where the container passed qc
vb_donors = get_visual_behavior_donors()['donor_id']
vb_sessions = not_failed.query("donor_id in @vb_donors")

from allensdk.brain_observatory.behavior import behavior_ophys_session as bos
from allensdk.brain_observatory.behavior import behavior_session as bs
from allensdk.brain_observatory.behavior import stimulus_processing
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.internal.api import behavior_lims_api as bla

from multiprocessing import  Pool
from functools import partial
import numpy as np

def parallelize(data, func, num_of_processes):

    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data

def run_on_subset(func, data_subset):
    return data_subset.apply(func, axis=1)

def parallelize_on_rows(data, func, num_of_processes=16):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)

def get_stage_name(row):
    if pd.isnull(row['ophys_experiment_id']):
        api = bla.BehaviorLimsApi(int(row['behavior_session_id']))
        session = bs.BehaviorSession(api)
    else:
        api = boa.BehaviorOphysLimsApi(int(row['ophys_experiment_id']))
        session = bos.BehaviorOphysSession(api)
    try:
        stage_name = api.get_task_parameters()['stage']
    except Exception:
        print("Load error")
        return "Load error"
    else:
        print(stage_name)
        return stage_name

df_to_use = vb_sessions.iloc[:20].copy()
#  df_to_use = vb_sessions
df_to_use['stage_name'] = parallelize_on_rows(df_to_use, get_stage_name)
#  df_to_use.to_hdf('/home/nick.ponvert/nco_home/data/vb_sessions.h5', key='df')

