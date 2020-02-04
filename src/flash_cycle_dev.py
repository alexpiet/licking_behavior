import psy_general_tools as pgt
import psy_timing_tools as pt
import matplotlib.pyplot as plt
import flash_cycle_tools as fct
import seaborn as sns
plt.ion()


# Licking wrt flash cycle
ids = pgt.get_active_ids()
directory = '/home/alex.piet/codebase/behavior/model_free/'

# Plot All Sessions Individually
fct.plot_all_sessions(ids,directory)

# Plot all Sessions together
fct.plot_sessions(ids,directory=directory+"all",return_counts=True)

# plot All sessions for each mouse
mice_ids = pgt.get_mice_ids()
fct.plot_all_mouse_sessions(mice_ids, directory)

# get dataframe of peakiness
df = fct.build_session_table(pgt.get_active_ids(),"/home/alex.piet/codebase/behavior/psy_fits_v9/")
plt.figure(); plt.plot(df.peakiness,df.hit_percentage,'ko');    plt.xlabel('PeakScore'); plt.ylabel('Hit Fraction'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_hit_fraction.png')
plt.figure(); plt.plot(df.peakiness,df.hit_count,'ko');         plt.xlabel('PeakScore'); plt.ylabel('Hit Count'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_hit_count.png')
plt.figure(); plt.plot(df.peakiness,df.licks,'ko');             plt.xlabel('PeakScore'); plt.ylabel('# Lick Bouts'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_lick_bouts.png')
plt.figure(); plt.plot(df.peakiness,df.task_index,'ko');        plt.xlabel('PeakScore'); plt.ylabel('Task/Timing Index'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_timing_task_index.png')
plt.figure(); plt.plot(df.peakiness,df.mean_dprime,'ko');       plt.xlabel('PeakScore'); plt.ylabel('mean dprime'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_mean_dprime.png')

plt.figure(); plt.plot(df.task_index,df.hit_percentage,'ko');   plt.xlabel('Task/Timing Index'); plt.ylabel('Hit Fraction'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_task_index_by_hit_fraction.png')
plt.figure(); plt.plot(df.task_index,df.hit_count,'ko');        plt.xlabel('Task/Timing Index'); plt.ylabel('Hit Count'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_task_index_hit_count.png')
plt.figure(); plt.plot(df.task_index,df.licks,'ko');            plt.xlabel('Task/Timing Index'); plt.ylabel('# Lick Bouts'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_task_index_lick_bouts.png')
plt.figure(); plt.plot(df.task_index,df.peakiness,'ko');        plt.xlabel('Task/Timing Index'); plt.ylabel('PeakScore'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_task_index_peak_score.png')
plt.figure(); plt.plot(df.task_index,df.mean_dprime,'ko');      plt.xlabel('Task/Timing Index'); plt.ylabel('mean dprime'); plt.savefig('/home/alex.piet/codebase/behavior/psy_fits_v9/all_peakiness_task_index_mean_dprime.png')


# Dev below here
#####################################################################
# Look at the start of lick bouts relative to flash cycle
all_licks = []
change_licks = []
for id in pgt.get_active_ids():
    print(id)
    try:
        session = pgt.get_data(id)
        pm.annotate_licks(session)
        pm.annotate_bouts(session)
        pm.annotate_bout_start_time(session)
        x = session.stimulus_presentations[session.stimulus_presentations['bout_start']==True]
        rel_licks = (x.bout_start_time-x.start_time).values
        all_licks.append(rel_licks)
        x = session.stimulus_presentations[(session.stimulus_presentations['bout_start']==True) & (session.stimulus_presentations['change'] ==True)]
        rel_licks = (x.bout_start_time-x.start_time).values
        change_licks.append(rel_licks)
    except Exception as e:
        print(" crash "+str(e))

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



