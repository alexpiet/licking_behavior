import psy_tools as ps
from alex_utils import *
import numpy as np

# save out 2x2 grid of blue/orange plots for each mouse
# calculate blue/orange ratio for each session
# Calculate post-ili, and pre-ili
# label licking bouts, and give them all the "rewarded" property
# label each lick as first of bout or last of bout
# Can look at ILI relative to start or end of bout

session_ids = ps.get_session_ids()
mice_ids = ps.get_mice_ids()
session = ps.get_data(session_ids[0])

sessions,IDS, active =ps.load_mouse(mice_ids[0])


