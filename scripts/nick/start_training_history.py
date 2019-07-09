import os
import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob 

python_file = r"/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/licking_behavior/scripts/nick/fit_training_history.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_jobs/20190708_fit_training_history'

job_settings = {'queue': 'braintv',
                'mem': '15g',
                'walltime': '24:00:00',
                'ppn':16,
                'jobdir': jobdir,
                }


#  All subjects
#  [795512663, 847074139, 847076515, 820871399, 823826963, 830896318,
#   834823464, 766015379, 830940312, 772629800, 784057617, 831004160,
#   830901414, 756674776, 772622642, 756577240, 760949537, 795522266,
#   744911447, 722884873, 791756316, 789992895, 842724844, 803258370,
#   843387577, 813703535, 814111925, 820878203]

#  subjects_to_use = [795512663]
subject_inds = [40, 41, 42, 43, 44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74,
75, 76, 77, 78, 79]

#  vb_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/vb_sessions.h5', key='df')
#  sessions_to_use = vb_sessions[vb_sessions['stage_name'] != 'Load error']
#  subjects_to_use = [847074139, 847076515, 820871399, 823826963, 830896318]
#  subject_sessions = sessions_to_use.query('donor_id in @subjects_to_use')

for row_loc in subject_inds:
    PythonJob(
        python_file,
        python_executable='/home/nick.ponvert/.conda/envs/visb/bin/python',
        python_args=row_loc,
        conda_env=None,
        jobname = 'licking_model_loc{}'.format(row_loc),
        **job_settings
    ).run(dryrun=False)
