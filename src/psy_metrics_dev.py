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
for id in ps.get_session_ids():
    print(id)
    try:
        filename = '/home/alex.piet/codebase/behavior/psy_fits_v2/'+str(id)
        session = ps.get_data(id)
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

lick_rates, reward_rates, all_epochs,times, count, all_times = pm.get_rates()
pm.plot_metrics(session)
pm.plot_all_epochs(all_epochs)                  # get epoch across time in session
pm.plot_all_times(times,count,all_times)        # get time in each epoch across all sessions
pm.plot_all_rates(lick_rates,reward_rates)
pm.plot_all_rates_averages(lick_rates,reward_rates)

# Population Summary Figures by stage
lick_ratesA, reward_ratesA, all_epochsA,timesA, countA, all_timesA = pm.get_rates(ids=ps.get_active_A_ids())
lick_ratesB, reward_ratesB, all_epochsB,timesB, countB, all_timesB = pm.get_rates(ids=ps.get_active_B_ids())
lick_rates1, reward_rates1, all_epochs1,times1, count1, all_times1 = pm.get_rates(ids=ps.get_stage_ids(1))
lick_rates3, reward_rates3, all_epochs3,times3, count3, all_times3 = pm.get_rates(ids=ps.get_stage_ids(3))
lick_rates4, reward_rates4, all_epochs4,times4, count4, all_times4 = pm.get_rates(ids=ps.get_stage_ids(4))
lick_rates6, reward_rates6, all_epochs6,times6, count6, all_times6 = pm.get_rates(ids=ps.get_stage_ids(6))

pm.compare_all_rates([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'])
pm.compare_all_rates_averages([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'])
pm.compare_all_rates([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'])
pm.compare_all_rates_averages([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'])
pm.compare_all_rates([lick_rates3,lick_rates4],[reward_rates3,reward_rates4],['3','4'])

pm.compare_all_times([timesA,timesB],[countA,countB],[all_timesA,all_timesB],['A','B'])
pm.compare_all_times([times1,times3,times4,times6],[count1,count3,count4,count6],[all_times1,all_times3,all_times4,all_times6],['1','3','4','6'])

pm.compare_all_epochs([all_epochsA,all_epochsB],['A','B'],smoothing=500)
pm.compare_all_epochs([all_epochs3,all_epochs4],['3','4'],smoothing=500) 

