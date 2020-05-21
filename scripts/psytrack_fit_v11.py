#!/usr/bin/env python

import sys
import psy_tools as ps

if __name__ == '__main__':
    name_of_this_file   = sys.argv[0]
    behavior_session_id       = sys.argv[1]
    
    dirc = "/home/alex.piet/codebase/behavior/psy_fits_v11/"
    ps.process_session(behavior_session_id,directory=dirc,format_options={'mean_center':True},complete=False,LATE_TASK=True)

