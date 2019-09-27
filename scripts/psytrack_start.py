import sys
#sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
#from pbstools import PythonJob
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/pbstools')
from pbstools import PythonJob 

python_file = r"/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/scripts/psytrack_fit.py"
jobdir = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v3/psytrack_20190927'
job_settings = {'queue': 'braintv',
                'mem': '15g',
                'walltime': '96:00:00',
                'ppn':1,
                'jobdir': jobdir,
                }

experiment_ids = [792813858, 792815735, 794381992, 795073741, 795076128, 795952471,
       795953296, 796105304, 796108483, 796308505, 797255551, 798404219,
       803736273, 805784331, 806455766, 806456687, 806989729, 807752719,
       807753318, 807753334, 808619543, 808621034, 808621958, 809497730,
       809501118, 811456530, 811458048, 813083478, 815652334, 817267785,
       820307518, 822647135, 825130141, 826587940, 830093338, 830700781,
       830700800, 833629926, 833631914, 834279496, 836258936, 836258957,
       836911939, 837296345, 837729902, 842973730, 843519218, 847125577,
       848692970, 848694639, 848697604, 848697625, 848698709, 849199228,
       849203565, 849203586, 850479305, 850489605, 851056106, 851060467,
       851932055, 852691524, 853328115, 853962951, 853962969, 854703305,
       855577488, 855582961, 855582981, 856096766, 859147033, 862848066,
       863735602, 864370674, 873972085, 877022592, 878363088, 879331157,
       880374622, 880961028]

for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/alex.piet/codebase/miniconda3/envs/visbeh/bin/python', 
        python_args=experiment_id,
        conda_env=None,
        jobname = 'psy_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)



