import psy_general_tools as pgt
import numpy as np
import psy_timing_tools as pt
from importlib import reload
import matplotlib.pyplot as plt
plt.ion()

session_ids = pgt.get_session_ids()
mice_ids = pgt.get_mice_ids()

## Plot Mean ILI over ophys sessions for all mice
all_durs = pt.get_all_mouse_durations(mice_ids)
pt.plot_all_mouse_durations(all_durs,directory='/home/alex.piet/codebase/behavior/model_free/')

# Plot Licks Distributions
pt.plot_all_mice_lick_distributions(mice_ids,directory='/home/alex.piet/codebase/behavior/model_free/')
pt.plot_all_session_lick_distributions(session_ids,directory='/home/alex.piet/codebase/behavior/model_free/')
pt.plot_lick_count(session_ids,directory='/home/alex.piet/codebase/behavior/model_free/')
pt.plot_bout_count(session_ids,directory='/home/alex.piet/codebase/behavior/model_free/')

# Plot Hazard Index Verification
dexes = pt.hazard_index(session_ids)  #### Crashing
pt.plot_hazard_index(dexes)

# Plot single session chronometric
session = pgt.get_data(session_ids[0])
pt.annotate_licks(session)
bout = pt.get_bout_table(session)
pt.get_chronometric(bout)

# Plot all chronometrics
pt.plot_all_session_chronometric(session_ids)
pt.plot_all_mice_chronometric(mice_ids)

# Plot single session licking bout verification
session = pgt.get_data(session_ids[0])
pt.annotate_licks(session)
pt.plot_session(session)

# Plot Bout ILI, and statistics for a single session
session = pgt.get_data(session_ids[0])
pt.annotate_licks(session)
bout = pt.get_bout_table(session)
pt.plot_bout_ili(bout, from_start=True,directory=directory+"example_")
pt.plot_bout_ili(bout, from_start=False,directory=directory+"example_")
pt.plot_bout_ili_current(bout, from_start=True,directory=directory+"example_")
pt.plot_bout_ili_current(bout, from_start=False,directory=directory+"example_")
pt.plot_bout_durations(bout,directory=directory+"example_")

# Plot Bout ILI, and statistics for a group of sessions
all_bout = pt.get_all_bout_table(session_ids)
durs = pt.get_all_bout_statistics(session_ids)
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory)
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory)
pt.plot_bout_ili(all_bout, from_start=False,directory=directory+"all_")
pt.plot_bout_ili(all_bout, from_start=True,directory=directory+"all_")
pt.plot_bout_ili_current(all_bout, from_start=True,directory=directory+"all_")
pt.plot_bout_ili_current(all_bout, from_start=False,directory=directory+"all_")
pt.plot_bout_durations(all_bout,directory=directory+"all_")

# Plot IBI by Stage
all_bout = pt.get_all_bout_table(pgt.get_active_A_ids())
durs = pt.get_all_bout_statistics(pgt.get_active_A_ids())
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"A_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"A_")

all_bout = pt.get_all_bout_table(pgt.get_active_B_ids())
durs = pt.get_all_bout_statistics(pgts.get_active_B_ids())
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"B_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"B_")

all_bout = pt.get_all_bout_table(pgt.get_stage_ids(1))
durs = pt.get_all_bout_statistics(pgt.get_stage_ids(1))
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"Stage1_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"Stage1_")

all_bout = pt.get_all_bout_table(pgt.get_stage_ids(3))
durs = pt.get_all_bout_statistics(pgt.get_stage_ids(3))
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"Stage3_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"Stage3_")

all_bout = pt.get_all_bout_table(pgt.get_stage_ids(4))
durs = pt.get_all_bout_statistics(pgt.get_stage_ids(4))
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"Stage4_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"Stage4_")

all_bout = pt.get_all_bout_table(pgt.get_stage_ids(6))
durs = pt.get_all_bout_statistics(pgt.get_stage_ids(6))
pt.plot_all_bout_statistics(durs, all_bout=all_bout,directory=directory+"Stage6_")
pt.plot_all_bout_statistics_current(durs, all_bout=all_bout,directory=directory+"Stage6_")




# Look at the start of lick bouts relative to flash cycle
all_licks = []
change_licks = []
for id in pgt.get_active_ids():
    print(id)
    try:
        session = pgt.get_data(id)
        pt.annotate_licks(session)
        pm.annotate_bouts(session)
        pm.annotate_bout_start_time(session)
        x = session.stimulus_presentations[session.stimulus_presentations['bout_start']==True]
        rel_licks = (x.bout_start_time-x.start_time).values
        all_licks.append(rel_licks)
        x = session.stimulus_presentations[(session.stimulus_presentations['bout_start']==True) & (session.stimulus_presentations['change'] ==True)]
        rel_licks = (x.bout_start_time-x.start_time).values
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

plt_all_licks(all_licks,change_licks,45)




