import psy_tools as ps
from alex_utils import *
import numpy as np
import psy_timing_tools as pt
from importlib import reload
import matplotlib.pyplot as plt
plt.ion()

session_ids = ps.get_session_ids()
mice_ids = ps.get_mice_ids()



## Plot Mean ILI over ophys sessions for all mice
all_durs = pt.get_all_mouse_durations(mice_ids)
pt.plot_all_mouse_durations(all_durs)

# Plot Licks Distributions
pt.plot_all_mice_lick_distributions(mice_ids)
pt.plot_all_session_lick_distributions(session_ids)
pt.plot_lick_count(session_ids)

# Plot Hazard Index Verification
dexes = pt.hazard_index(session_ids)
pt.plot_hazard_index(dexes)

# Plot single session chronometric
session = ps.get_data(session_ids[0])
pt.annotate_licks(session)
bout = pt.get_bout_table(session)
pt.get_chronometric(bout)

# Plot all chronometrics
pt.plot_all_session_chronometric(session_ids)
pt.plot_all_mice_chronometric(mice_ids)

# Plot single session licking bout verification
session = ps.get_data(session_ids[0])
pt.annotate_licks(session)
pt.plot_session(session)

# Plot Bout ILI, and statistics for a single session
session = ps.get_data(session_ids[0])
pt.annotate_licks(session)
bout = pt.get_bout_table(session)
pt.plot_bout_ili(bout, from_start=T/F)
pt.plot_bout_ili_current(bout, from_start=T/F)
pt.plot_bout_durations(bout)

# Plot Bout ILI, and statistics for a group of sessions
all_bout = pt.get_all_bout_table(session_ids)
durs = pt.get_all_bout_statistics(session_ids)
pt.plot_all_bout_statistics(durs, all_bout=all_bout)
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout)


# Look at the start of lick bouts relative to flash cycle
all_licks = []
change_licks = []
for id in ps.get_active_ids():
    print(id)
    try:
        session = ps.get_data(id)
        pt.annotate_licks(session)
        pm.annotate_bouts(session)
        x = session.stimulus_presentations[session.stimulus_presentations['bout_start']==True]
        rel_licks = np.concatenate((x.licks-x.start_time).values)
        all_licks.append(rel_licks)
        x = session.stimulus_presentations[(session.stimulus_presentations['bout_start']==True) & (session.stimulus_presentations['change'] ==True)]
        rel_licks = np.concatenate((x.licks-x.start_time).values)
        change_licks.append(rel_licks)
    except:
        print(" crash")

def plt_all_licks(all_licks,change_licks,bins):
    plt.figure()
    plt.hist(np.concatenate(all_licks),bins=bins,color='gray',label='All Flashes')
    plt.hist(np.concatenate(change_licks),bins=bins,color='black',label='Change Flashes')
    plt.ylabel('Count',fontsize=12)
    plt.xlabel('Time since last flash onset',fontsize=12)
    plt.xlim([0, 0.75])
    plt.legend()
    plt.tight_layout()

plt_all_licks(all_licks,change_licks,50)


