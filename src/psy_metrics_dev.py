import numpy as np
import psy_tools as ps
import matplotlib.pyplot as plt
import psy_timing_tools as pt
import psy_metrics_tools as pm
import pandas as pd
plt.ion()

id = ps.get_session_ids()[27] #30

session = ps.get_data(id)
fit = ps.load_fit(session.ophys_experiment_id)
fit = ps.plot_fit(session.ophys_experiment_id,cluster_labels=fit['all_clusters']['3'][1],num_clusters=3)
fit = ps.plot_fit(session.ophys_experiment_id,cluster_labels=fit['cluster']['3'][1],num_clusters=3)


df = session.get_rolling_performance_df()
trials = session.trials
trials = pd.concat([trials,df],axis=1,sort=False)


# Plot rolling indices
plt.figure()
plt.plot(trials.start_time, trials.reward_rate,'k',label='reward')
plt.plot(trials.start_time, trials.rolling_dprime,'b',label='dprime')
plt.plot(trials.start_time, trials.hit_rate,'r',label='hit')

# make these plots for all sessions
for id in ps.get_session_ids():
    print(id)
    try:
        filename = '/home/alex.piet/codebase/behavior/psy_fits_v2/'+str(id)
        session = ps.get_data(id)
        pt.annotate_licks(session)
        pm.annotate_bouts(session)
        pm.annotate_flash_rolling_metrics(session)
        pm.classify_by_flash_metrics(session)
        pm.plot_metrics(session,filename=filename+'_metrics')
        pm.plot_2D(session,filename=filename+'_metrics_2D')
        plt.close('all')
    except:
        print(' crash')



