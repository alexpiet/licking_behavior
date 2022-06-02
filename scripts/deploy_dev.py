import os
import time
import argparse
from simple_slurm import Slurm
import psy_general_tools as pgt

# Parse optional arguments
parser = argparse.ArgumentParser(description='deploy behavior fits to cluster')
parser.add_argument('--env-path', type=str, default='visbeh', metavar='path to conda environment to use')
parser.add_argument('--version', type=str, default='0', metavar='model version')
parser.add_argument(
    '--force-overwrite', 
    action='store_true',
    default=False,
    dest='force_overwrite', 
    help='Overwrites existing fits for this version if enabled. Otherwise only sessions without existing results are fit'
)

if __name__ == "__main__":
    
    # Determine python version to use
    args = parser.parse_args()
    python_executable = "{}/bin/python".format(args.env_path)
    print('python executable = {}'.format(python_executable))
    python_file = "/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/licking_behavior/scripts/dev.py"

    # Define output for logs
    stdout_basedir = "/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/logs"
    stdout_location = os.path.join(stdout_basedir, 'job_records_dev_{}'.format(args.version))
    if not os.path.exists(stdout_location):
        print('making folder {}'.format(stdout_location))
        os.mkdir(stdout_location)
    print('stdout files will be at {}'.format(stdout_location))
    
    # Get list of behavior_session_ids
    manifest = pgt.get_ophys_manifest()
    behavior_session_ids = manifest.behavior_session_id.values
    print('behavior_session_ids loaded')

    # Iterate through sessions and start jobs if needed
    job_count = 0
    job_string = "--bsid {} --version {}"
    print('Starting model version '+str(args.version))
    for behavior_session_id in behavior_session_ids:    
        # Set up job
        job_count += 1
        print('starting cluster job for {}, job count = {}'.format(behavior_session_id, job_count))
        job_title = 'oeid_{}_beh_v_{}'.format(behavior_session_id, args.version)
        walltime = '1:00:00'
        mem = '2gb'
        job_id = Slurm.JOB_ARRAY_ID
        job_array_id = Slurm.JOB_ARRAY_MASTER_ID
        output = stdout_location+"/"+str(job_array_id)+"_"+str(job_id)+"_"+str(behavior_session_id)+".out"
    
        # instantiate a SLURM object
        slurm = Slurm(
            cpus_per_task=4,
            job_name=job_title,
            time=walltime,
            mem=mem,
            output= output,
            partition="braintv"
        )

        # Start job
        args_string = job_string.format(behavior_session_id, args.version)
        slurm.sbatch('{} {} {}'.format(
                python_executable,
                python_file,
                args_string,
            )
        )
        time.sleep(0.001)
