import numpy as np
import psy_general_tools as pgt
import matplotlib.pyplot as plt
import psy_metrics_tools as pm
import pandas as pd
plt.ion()

# Basic Example
id = pgt.get_session_ids()[27]  # Pick Session
session = pgt.get_data(id)      # Get SDK session object
pm.get_metrics(session)         # annotate session
pm.plot_metrics(session)        # plots metrics for this session
pm.plot_2D(session)             # Plots licking rate vs reward rate
durations = pm.get_time_in_epochs(session) # Duration of each epoch

# make session plots for all sessions
for id in pgt.get_active_ids():
    print(id)
    try:
        filename = '/home/alex.piet/codebase/behavior/model_free/'+str(id)
        session = pgt.get_data(id)
        pm.get_metrics(session)
        pm.plot_metrics(session,filename=filename+'_metrics')
        pm.plot_2D(session,filename=filename+'_metrics_2D')
        plt.close('all')
    except:
        print(' crash')

# Population Summary Figures
import seaborn as sns
sns.set_context('notebook', font_scale=1, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': False, 'ytick.left': False,})

lick_rates, reward_rates, all_epochs,times, count, all_times,all_hit_fraction,all_hit_rate,all_fa_rate,all_dprime = pm.get_rates()
pm.plot_metrics(session)
pm.plot_all_epochs(all_epochs)                  # get epoch across time in session
pm.plot_all_times(times,count,all_times)        # get time in each epoch across all sessions
pm.plot_all_rates(lick_rates,reward_rates)
pm.plot_all_rates_averages(lick_rates,reward_rates)
pm.compare_all_rates_averages([lick_rates],[reward_rates],rlabels=['All'],split_on=2400)
pm.plot_all_dprime(all_dprime)
pm.plot_all_performance_rates(all_hit_fraction,all_hit_rate,all_fa_rate)
pm.plot_all_performance_rates_averages(all_dprime,all_hit_fraction,all_hit_rate,all_fa_rate)
pm.compare_all_performance_rates_averages([all_dprime],[all_hit_fraction],[all_hit_rate],[all_fa_rate],['All'],split_on=2400)
num_hits = pm.get_num_hits(ps.get_active_ids())


# by cre line
lick_rates_slc, reward_rates_slc, all_epochs_slc,times_slc, count_slc, all_times_slc,all_hit_fraction_slc,all_hit_rate_slc,all_fa_rate_slc,all_dprime_slc = pm.get_rates(ids = ps.get_slc_session_ids())
lick_rates_vip, reward_rates_vip, all_epochs_vip,times_vip, count_vip, all_times_vip,all_hit_fraction_vip,all_hit_rate_vip,all_fa_rate_vip,all_dprime_vip = pm.get_rates(ids = ps.get_vip_session_ids())
pm.compare_all_rates([lick_rates_slc,lick_rates_vip],[reward_rates_slc,reward_rates_vip],['SLC','VIP'])
pm.compare_all_rates_averages([lick_rates_slc,lick_rates_vip],[reward_rates_slc,reward_rates_vip],['SLC','VIP'])
pm.compare_all_dprime([all_dprime_slc,all_dprime_vip],['SLC','VIP'])
pm.compare_all_performance_rates([all_hit_fraction_slc,all_hit_fraction_vip],[all_hit_rate_slc,all_hit_rate_vip],[all_fa_rate_slc,all_fa_rate_vip],['SLC','VIP'])
pm.compare_all_performance_rates_averages([all_dprime_slc,all_dprime_vip],[all_hit_fraction_slc,all_hit_fraction_vip],[all_hit_rate_slc,all_hit_rate_vip],[all_fa_rate_slc,all_fa_rate_vip],['SLC','VIP'])



# Population Summary Figures by stage
lick_ratesA, reward_ratesA, all_epochsA,timesA, countA, all_timesA, all_hfA, all_hrA, all_faA, all_dprimeA = pm.get_rates(ids=pgt.get_active_A_ids())
lick_ratesB, reward_ratesB, all_epochsB,timesB, countB, all_timesB, all_hfB, all_hrB, all_faB, all_dprimeB = pm.get_rates(ids=pgt.get_active_B_ids())
lick_rates1, reward_rates1, all_epochs1,times1, count1, all_times1, all_hf1, all_hr1, all_fa1, all_dprime1 = pm.get_rates(ids=pgt.get_stage_ids(1))
lick_rates3, reward_rates3, all_epochs3,times3, count3, all_times3, all_hf3, all_hr3, all_fa3, all_dprime3 = pm.get_rates(ids=pgt.get_stage_ids(3))
lick_rates4, reward_rates4, all_epochs4,times4, count4, all_times4, all_hf4, all_hr4, all_fa4, all_dprime4 = pm.get_rates(ids=pgt.get_stage_ids(4))
lick_rates6, reward_rates6, all_epochs6,times6, count6, all_times6, all_hf6, all_hr6, all_fa6, all_dprime6 = pm.get_rates(ids=pgt.get_stage_ids(6))
num_hitsA = pm.get_num_hits(ps.get_active_A_ids())
num_hitsB = pm.get_num_hits(ps.get_active_B_ids())
num_hits1 = pm.get_num_hits(ps.get_stage_ids(1))
num_hits3 = pm.get_num_hits(ps.get_stage_ids(3))
num_hits4 = pm.get_num_hits(ps.get_stage_ids(4))
num_hits6 = pm.get_num_hits(ps.get_stage_ids(6))

from scipy import stats
import seaborn as sns

#num_hitsA = np.array(num_hitsA)[np.array(num_hitsA) > 50]
#num_hitsB = np.array(num_hitsB)[np.array(num_hitsB) > 50]
pm.compare_hit_count(num_hitsA,num_hitsB)
pm.compare_all_performance_rates_averages_dprime([all_dprimeA,all_dprimeB],['A','B'])
pm.compare_all_performance_rates_averages_dprime([all_dprimeA,all_dprimeB],['A','B'],split_on=2400)




# A v B
pm.compare_all_rates([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'])
pm.compare_all_rates_averages([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'])
pm.compare_all_rates_averages([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],split_on=2400)
pm.compare_all_dprime([all_dprimeA,all_dprimeB],['A','B'])
pm.compare_all_performance_rates([all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'])
pm.compare_all_performance_rates_averages([all_dprimeA,all_dprimeB],[all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'])
pm.compare_all_performance_rates_averages([all_dprimeA,all_dprimeB],[all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],split_on=2400)


# by stage
pm.compare_all_rates([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'])
pm.compare_all_rates_averages([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'])
pm.compare_all_rates_averages([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'],split_on=2400)
pm.compare_all_dprime([all_dprime1,all_dprime3,all_dprime4,all_dprime6],['1','3','4','6'])
pm.compare_all_performance_rates([all_hf1,all_hf3,all_hf4,all_hf6],[all_hr1,all_hr3,all_hr4,all_hr6],[all_fa1,all_fa3,all_fa4,all_fa6],['1','3','4','6'])
pm.compare_all_performance_rates_averages([all_dprime1,all_dprime3,all_dprime4,all_dprime6],[all_hf1,all_hf3,all_hf4,all_hf6],[all_hr1,all_hr3,all_hr4,all_hr6],[all_fa1,all_fa3,all_fa4,all_fa6],['1','3','4','6'])
pm.compare_all_performance_rates_averages([all_dprime1,all_dprime3,all_dprime4,all_dprime6],[all_hf1,all_hf3,all_hf4,all_hf6],[all_hr1,all_hr3,all_hr4,all_hr6],[all_fa1,all_fa3,all_fa4,all_fa6],['1','3','4','6'],split_on=2400)


# Just 3 v 4
pm.compare_all_rates([lick_rates3,lick_rates4],[reward_rates3,reward_rates4],['3','4'])
pm.compare_all_dprime([all_dprime3,all_dprime4],['3','4'])
pm.compare_all_performance_rates([all_hf3,all_hf4],[all_hr3,all_hr4],[all_fa3,all_fa4],['3','4'])

# Clustering
pm.compare_all_times([timesA,timesB],[countA,countB],[all_timesA,all_timesB],['A','B'])
pm.compare_all_times([times1,times3,times4,times6],[count1,count3,count4,count6],[all_times1,all_times3,all_times4,all_times6],['1','3','4','6'])
pm.compare_all_epochs([all_epochsA,all_epochsB],['A','B'],smoothing=500)
pm.compare_all_epochs([all_epochs3,all_epochs4],['3','4'],smoothing=500) 

