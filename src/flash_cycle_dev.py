import psy_general_tools as pgt
import psy_timing_tools as pt
import matplotlib.pyplot as plt
import flash_cycle_tools as fct
import seaborn as sns
plt.ion()


# Licking wrt flash cycle
ids = pgt.get_active_ids()
dir = '/home/alex.piet/codebase/behavior/model_free/'
fct.plot_session(ids[0])


# Plot All Sessions Individually
for id in ids:
    print(id)
    fct.plot_session(id,directory=dir+str(id))
    plt.close('all')

# Plot all Sessions together
fct.plot_sessions(ids,directory=dir+"all")

# plot All sessions for each mouse
mice_ids = pgt.get_mice_ids()
for mouse in mice_ids:
    print(mouse)
    mice_ids = pgt.get_mice_sessions(mouse)
    fct.plot_sessions(mice_ids, directory=dir+"Mouse_"+str(mouse))

# get dataframe of peakiness
df = fct.build_session_table(pgt.get_active_ids(),"/home/alex.piet/codebase/behavior/psy_fits_v2/")
plt.figure(); plt.plot(df.peakiness,df.hit_percentage,'ko'); plt.xlabel('PeakScore'); plt.ylabel('Hit Fraction')
plt.figure(); plt.plot(df.peakiness,df.hit_count,'ko'); plt.xlabel('PeakScore'); plt.ylabel('Hit Count')
plt.figure(); plt.plot(df.peakiness,df.licks,'ko'); plt.xlabel('PeakScore'); plt.ylabel('# Lick Bouts')
plt.figure(); plt.plot(df.peakiness,df.task_index,'ko'); plt.xlabel('PeakScore'); plt.ylabel('Timing/Task Index')

plt.figure(); plt.plot(df.task_index,df.hit_percentage,'ko');   plt.xlabel('Timing/Task Index'); plt.ylabel('Hit Fraction')
plt.figure(); plt.plot(df.task_index,df.hit_count,'ko');        plt.xlabel('Timing/Task Index'); plt.ylabel('Hit Count')
plt.figure(); plt.plot(df.task_index,df.licks,'ko');            plt.xlabel('Timing/Task Index'); plt.ylabel('# Lick Bouts')
plt.figure(); plt.plot(df.task_index,df.peakiness,'ko');        plt.xlabel('Timing/Task Index'); plt.ylabel('PeakScore')


