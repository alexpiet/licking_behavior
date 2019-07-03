import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob 
python_file = r"/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/scripts/20190703_psytrack_fit.py"
jobdir = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits/20190703_psytrack'
job_settings = {'queue': 'braintv',
                'mem': '15g',
                'walltime': '24:00:00',
                'ppn':1,
                'jobdir': jobdir,
                }

from get_sdk_ids import get_ids
experiment_ids = get_ids()
 
for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/alex.piet/codebase/miniconda3/envs/visbeh/bin/python', 
        python_args=experiment_id,
        conda_env=None,
        jobname = 'psy_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)



