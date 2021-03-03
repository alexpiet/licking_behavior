#!/usr/bin/env python

import sys
import psy_tools as ps

if __name__ == '__main__':
    name_of_this_file   = sys.argv[0]
    behavior_session_id       = sys.argv[1]
    ps.process_session(behavior_session_id,version=20,format_options={'mean_center':True})

