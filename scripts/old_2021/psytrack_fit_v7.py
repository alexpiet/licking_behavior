#!/usr/bin/env python

import sys
import psy_tools as ps

if __name__ == '__main__':
    name_of_this_file   = sys.argv[0]
    experiment_id       = sys.argv[1]
    
    dirc = "/home/alex.piet/codebase/behavior/psy_fits_v7/"
    ps.process_session(experiment_id,directory=dirc)

