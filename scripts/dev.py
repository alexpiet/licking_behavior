import psy_general_tools as pgt
import psy_metrics_tools as pm
import argparse

parser = argparse.ArgumentParser(description='fit behavioral model for session_id')
parser.add_argument(
    '--bsid', 
    type=int, 
    default=0,
    metavar='bsid',
    help='behavior session id'
)
parser.add_argument(
    '--version', 
    type=str, 
    default='',
    metavar='behavior model version',
    help='model version to use'
)

if __name__ == '__main__':
    args = parser.parse_args()
    print(args.bsid)
    session = pgt.get_data(args.bsid)
    pm.licks_each_image(session)
    pm.rewards_each_image(session)
    pm.annotate_licks(session)
    pm.annotate_bouts(session)
    pm.annotate_flash_rolling_metrics(session)
    # Check the log file and use 'egrep -lir "error" '
