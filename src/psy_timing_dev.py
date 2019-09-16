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
