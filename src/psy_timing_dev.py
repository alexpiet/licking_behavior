import psy_tools as ps
from alex_utils import *
import numpy as np
import psy_timing_tools as pt
from importlib import reload
import matplotlib.pyplot as plt
plt.ion()

# calculate blue/orange ratio for each session
# Can look at ILI relative to start or end of bout
# Look at ILI for rewarded vs non rewarded pre bout

session_ids = ps.get_session_ids()
mice_ids = ps.get_mice_ids()
session = ps.get_data(session_ids[0])
sessions,IDS, active =ps.load_mouse(mice_ids[0])

## Plot Summary
all_durs = pt.get_all_mouse_durations(mice_ids)
pt.plot_all_mouse_durations(all_durs)

# Plot Distribution plots
mice_ids = ps.get_mice_ids()
directory = '/home/alex.piet/codebase/behavior/psy_fits_v2/'
for mouse in mice_ids:
    print(mouse)
    try:
        pt.plot_mouse_lick_distributions(mouse,directory=directory)
    except:
        print(" crash")

total = []
hits = []
for id in session_ids:
    print(id)
    try:
        this_total,this_hit = pt.get_lick_count(id)
        total.append(this_total)
        hits.append(this_hit)
    except:
        print(" crash")

plt.figure()
plt.plot(total,hits,'ko')
plt.ylabel('# Hits',fontsize=12)
plt.xlabel('# Non-bout Licks',fontsize=12)



for id in session_ids:
    print(id)
    try:
        pt.plot_lick_distribution(ps.get_data(id),directory=directory)
    except:
        print(" crash")








