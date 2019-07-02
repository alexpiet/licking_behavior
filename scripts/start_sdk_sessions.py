import os
import sys
sys.path.append('/allen/programs/braintv/workgroups/nc-ophys/Doug/pbstools')
from pbstools import PythonJob 

python_file = r"/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/src/licking_behavior/scripts/fit_sdk_sessions.py"

jobdir = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/cluster_jobs/20190629_sdk_fit'

job_settings = {'queue': 'braintv',
                'mem': '15g',
                'walltime': '24:00:00',
                'ppn':1,
                'jobdir': jobdir,
                }

# with subjects_to_use = [847076515]
# experiment_ids = [891996193, 893831526, 891054695, 892805315, 889772922]

# with subjects_to_use = [827778598, 814111925, 813703535, 820878203, 803258370]
#  experiment_ids = [862023618, 862023618, 862848084, 862848084, 844395446, 842973730,
#         843519218, 842975542, 847241639, 843520488, 848007790, 847125577,
#         851060467, 850479305, 850489605, 845037476, 849199228, 846487947,
#         849203565, 848692970, 849204593, 848694045, 851932055, 848694639,
#         851056106, 854703904, 866463736, 866463736, 848697625, 853328133,
#         848698709, 853962969, 855582961, 852691524, 868911434, 868911434,
#         869972431, 869972431, 871159631, 871159631]


# with subjects_to_use = [800311507, 789992895, 772629800, 766025752, 766015379, 788982905,
# 843387577, 789014546, 830940312, 834902192, 820871399, 827778598]
experiment_ids = [820298614, 813083478, 813070010, 825615139, 888666715, 820307042,
       820307518, 822656725, 815652334, 822641265, 817267785, 862848066,
       822015264, 822647135, 878363070, 823372519, 806456687, 822028017,
       817267860, 822647116, 821011078, 825120601, 825130141, 823396897,
       823392290, 822024770, 840702910, 862023618, 862023618, 841948542,
       826576503, 826585773, 862848084, 862848084, 827230913, 826583436,
       827236946, 831330404, 825623170, 846490568, 849203586, 830697288,
       848694025, 866463736, 866463736, 848697604, 855582981, 868911434,
       868911434, 856096766, 869972431, 869972431, 829408506, 817251835,
       871159631, 871159631, 877696762, 864370674, 865744231, 879332693,
       807752719, 816795311, 880375092, 889777243, 810120743, 808619543,
       811456530, 884218326, 882935355, 885061426, 884221469, 885067826,
       885933191]


for experiment_id in experiment_ids:
    PythonJob(
        python_file,
        python_executable='/home/nick.ponvert/.conda/envs/visb/bin/python',
        python_args=experiment_id,
        conda_env=None,
        jobname = 'licking_model_{}'.format(experiment_id),
        **job_settings
    ).run(dryrun=False)
