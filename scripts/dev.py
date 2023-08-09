import psy_output_tools as po
import psy_general_tools as pgt
import psy_tools as ps
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
    session_df = ps.load_session_strategy_df(args.bsid, args.version)
    #session_df = ps.add_reward_time_to_session_df(session,session_df)
    session_df = ps.add_running_speed_to_session_df(session, session_df)
    session_df.to_csv(pgt.get_directory(args.version, \
        subdirectory='strategy_df')+str(args.bsid)+'.csv') 
    print('Finished')
    # Check the log file and use 'egrep -lir "error" '
    # Or egrep -Lir "Finished"
