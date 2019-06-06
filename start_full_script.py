import os
import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob 
python_file = r"/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/licking_behavior/model_fitting_script.py"
jobdir = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_jobs/vba_refit'
job_settings = {'queue': 'braintv',
                'mem': '15g',
                'walltime': '12:00:00',
                'ppn':1,
                'jobdir': jobdir,
                }

experiment_ids = [837729902, 838849930,836910438,840705705,840157581,841601446,840702910,841948542,841951447,842513687,842973730,843519218,846490568,847125577,848697604] # start jobs, one for each of these experiment IDS
for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/alex.piet/codebase/miniconda3/envs/visbeh/bin/python', # path to conda environment that has the correct python version, and all needed packages
        python_args=experiment_id,
        conda_env=None,
        jobname = 'full_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)



