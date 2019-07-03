import sys
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
#from pbstools import PythonJob
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob 

python_file = r"/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/scripts/20190703_psytrack_fit.py"
jobdir = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits/20190703_psytrack'
job_settings = {'queue': 'braintv',
                'mem': '15g',
                'walltime': '24:00:00',
                'ppn':1,
                'jobdir': jobdir,
                }

#from get_sdk_ids import get_ids
#experiment_ids = get_ids()
experiment_ids = [820298614, 813083478, 813070010, 825615139, 888666715, 820307042,
       820307518, 822656725, 815652334, 822641265, 817267785, 862848066,
       822015264, 822647135, 878363070, 823372519, 806456687, 822028017,
       817267860, 822647116, 821011078, 825120601, 825130141, 823396897,
       823392290, 822024770, 840702910, 862023618, 841948542, 826576503,
       826585773, 862848084, 827230913, 826583436, 827236946, 831330404,
       825623170, 846490568, 849203586, 830697288, 896160394, 848694025,
       866463736, 848697604, 855582981, 868911434, 856096766, 869972431,
       829408506, 817251835, 871159631, 877696762, 864370674, 865744231,
       879332693, 807752719, 816795311, 880375092, 889777243, 810120743,
       808619543, 811456530, 884218326, 882935355, 885061426, 884221469,
       885067826, 885933191]

for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/alex.piet/codebase/miniconda3/envs/visbeh/bin/python', 
        python_args=experiment_id,
        conda_env=None,
        jobname = 'psy_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)



