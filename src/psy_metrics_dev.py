import numpy as np
import psy_tools as ps
import matplotlib.pyplot as plt
import psy_timing_tools as pt
import psy_metrics_tools as pm
import pandas as pd
plt.ion()

id = ps.get_session_ids()[27] #30
session = ps.get_data(id)

# Get model clusters
fit = ps.load_fit(session.ophys_experiment_id)
fit = ps.plot_fit(session.ophys_experiment_id,cluster_labels=fit['all_clusters']['3'][1],num_clusters=3)
fit = ps.plot_fit(session.ophys_experiment_id,cluster_labels=fit['cluster']['3'][1],num_clusters=3)

# Plot rolling indices
df = session.get_rolling_performance_df()
trials = session.trials
trials = pd.concat([trials,df],axis=1,sort=False)
plt.figure()
plt.plot(trials.start_time, trials.reward_rate,'k',label='reward')
plt.plot(trials.start_time, trials.rolling_dprime,'b',label='dprime')
plt.plot(trials.start_time, trials.hit_rate,'r',label='hit')

# make metric plots for all sessions
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

lick_rates, reward_rates, all_epochs,times, count, all_times = pm.get_rates()
# get epoch across time in session
pm.plot_all_epochs(all_epochs)
# get time in each epoch across all sessions
pm.plot_all_times(times,count,all_times)
# Get summary plots in terms of rates
pm.plot_all_rates(lick_rates,reward_rates)
pm.plot_all_rates_averages(lick_rates,reward_rates)

# SFN figures
import seaborn as sns
sns.set_context('notebook', font_scale=1, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': False, 'ytick.left': False,})

pm.plot_metrics(session)
pm.plot_all_epochs(all_epochs)
pm.plot_all_times(times,count,all_times)
pm.plot_all_rates(lick_rates,reward_rates)
pm.plot_all_rates_averages(lick_rates,reward_rates)


