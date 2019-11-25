#!/usr/bin/env python

import sys
import psy_tools as ps

if __name__ == '__main__':
    name_of_this_file   = sys.argv[0]
    donor_id       = sys.argv[1]
    
    directory = '/home/alex.piet/codebase/behavior/psy_fits_v6/'
    ps.process_mouse(donor_id,directory=directory)

