import psy_general_tools as pgt
import numpy as np
import psy_timing_tools as pt
import psy_metrics_tools as pm

# TODO, Issue #234
# Plot the ILI distribution for a session
pt.plot_session_interlick_interval_distribution(session)
pt.plot_all_session_interlick_distributions(summary_df,version)

# TODO, Issue #234
## Plot Mean ILI over ophys sessions for all mice
all_durs = pt.get_all_mouse_durations(mice_ids)
pt.plot_all_mouse_durations(all_durs,directory='/home/alex.piet/codebase/behavior/model_free/')

# TODO, Issue #231
# Plot Hazard Index Verification ### NEED TO RUN STILL
dexes = ps.hazard_index(session_ids) 
ps.plot_hazard_index(dexes)

# TODO, Issue #231
# Plot single session chronometric
session = pgt.get_data(session_ids[0])
pm.annotate_licks(session)
bout = pt.get_bout_table(session)
pt.get_chronometric(bout) 

# TODO, Issue #231
# Plot all chronometrics
pt.plot_all_session_chronometric(session_ids)
pt.plot_all_mice_chronometric(mice_ids)

# TODO, Issue #232
# Plot single session licking bout verification
session = pgt.get_data(session_ids[0])
pm.annotate_licks(session)
pt.plot_session(session)

# TODO, Issue #233
# Plot Bout ILI, and statistics for a single session
session = pgt.get_data(session_ids[100])
pm.annotate_licks(session)
bout = pt.get_bout_table(session)
pt.plot_bout_ili(bout, from_start=True,directory=directory+"example_")
pt.plot_bout_ili(bout, from_start=False,directory=directory+"example_")
pt.plot_bout_ili_current(bout, from_start=True,directory=directory+"example_")
pt.plot_bout_ili_current(bout, from_start=False,directory=directory+"example_")
pt.plot_bout_durations(bout,directory=directory+"example_")

# TODO, Issue #233
# Plot Bout ILI, and statistics for a group of sessions
if False:
    all_bout = pt.get_all_bout_table(session_ids)
    durs = pt.get_all_bout_statistics(session_ids)
else:
    import psy_tools as ps
    d = ps.load('/home/alex.piet/codebase/behavior/data/psy_timing_all_bout_statistics_01_27_2020.pkl')
    durs = d['durs']
    all_bout = d['all_bout']

# TODO, Issue #233
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory)
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory)
pt.plot_bout_ili(all_bout, from_start=False,directory=directory+"all_")
pt.plot_bout_ili(all_bout, from_start=True,directory=directory+"all_")
pt.plot_bout_ili_current(all_bout, from_start=True,directory=directory+"all_")
pt.plot_bout_ili_current(all_bout, from_start=False,directory=directory+"all_")
pt.plot_bout_durations(all_bout,directory=directory+"all_")

# TODO, Issue #233
# Plot IBI by Stage
all_bout = pt.get_all_bout_table(pgt.get_active_A_ids())
durs = pt.get_all_bout_statistics(pgt.get_active_A_ids())
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"A_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"A_")

# TODO, Issue #233
all_bout = pt.get_all_bout_table(pgt.get_active_B_ids())
durs = pt.get_all_bout_statistics(pgt.get_active_B_ids())
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"B_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"B_")

# TODO, Issue #233
all_bout = pt.get_all_bout_table(pgt.get_stage_ids(1))
durs = pt.get_all_bout_statistics(pgt.get_stage_ids(1))
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"Stage1_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"Stage1_")

# TODO, Issue #233
all_bout = pt.get_all_bout_table(pgt.get_stage_ids(3))
durs = pt.get_all_bout_statistics(pgt.get_stage_ids(3))
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"Stage3_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"Stage3_")

# TODO, Issue #233
all_bout = pt.get_all_bout_table(pgt.get_stage_ids(4)) #update
durs = pt.get_all_bout_statistics(pgt.get_stage_ids(4))
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"Stage4_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"Stage4_")

# TODO, Issue #233
all_bout = pt.get_all_bout_table(pgt.get_stage_ids(6))#update
durs = pt.get_all_bout_statistics(pgt.get_stage_ids(6))
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"Stage6_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"Stage6_")


