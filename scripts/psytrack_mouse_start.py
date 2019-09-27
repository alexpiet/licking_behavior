import sys
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
#from pbstools import PythonJob
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob 

python_file = r"/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/scripts/psytrack_mouse_fit.py"
jobdir = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v3/psytrack_mouse_20190927'
job_settings = {'queue': 'braintv',
                'mem': '15g',
                'walltime': '48:00:00',
                'ppn':1,
                'jobdir': jobdir,
                }


experiment_ids = [744911447,756674776,760949537,772622642,772629800,784057617,789992895,791756316,803258370,813703535,820871399,820878203,823826963,834823464]

for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/alex.piet/codebase/miniconda3/envs/visbeh/bin/python', 
        python_args=experiment_id,
        conda_env=None,
        jobname = 'psym_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)



