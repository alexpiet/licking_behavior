import numpy as np
import psy_tools as ps
import matplotlib.pyplot as plt
import psy_timing_tools as pt
import pandas as pd
plt.ion()

id = ps.get_session_ids()[27] #30

session = ps.get_data(id)
fit = ps.plot_fit(session.ophys_experiment_id)

df = session.get_rolling_performance_df()
fit = ps.plot_fit(session.ophys_experiment_id)
pt.annotate_licks(session)
trials = session.trials
trials = pd.concat([trials,df],axis=1,sort=False)

# Get Bout Rate / second
bout_starts = session.licks[session.licks['bout_start']]
session.stimulus_presentations['bout_start'] = False
for index,x in bout_starts.iterrows():
    session.stimulus_presentations.at[session.stimulus_presentations[session.stimulus_presentations['start_time'].gt(x.timestamps)].index[0]-1,'bout_start'] = True

session.stimulus_presentations.drop(-1,inplace=True)
session.stimulus_presentations['bout_rate'] = session.stimulus_presentations['bout_start'].rolling(320,min_periods=1, win_type='triang').mean()/.75

# Get Lick Rate / second
session.stimulus_presentations['licked'] = [1 if len(this_lick) > 0 else 0 for this_lick in session.stimulus_presentations['licks']]
session.stimulus_presentations['lick_rate'] = session.stimulus_presentations['licked'].rolling(320, win_type='triang').mean()/.75

# Get Reward Rate / second
session.stimulus_presentations['rewarded'] = [1 if len(this_reward) > 0 else 0 for this_reward in session.stimulus_presentations['rewards']]
session.stimulus_presentations['reward_rate'] = session.stimulus_presentations['rewarded'].rolling(320,min_periods=1,win_type='triang').mean()/.75

session.stimulus_presentations['running_rate'] = session.stimulus_presentations['mean_running_speed'].rolling(320,min_periods=1,win_type='triang').mean()/.75

# Plot rolling indices
#plt.figure()
#plt.plot(trials.start_time, trials.reward_rate,'k',label='reward')
#plt.plot(trials.start_time, trials.rolling_dprime,'b',label='dprime')
#plt.plot(trials.start_time, trials.hit_rate,'r',label='hit')

plt.figure()
stim = psd.add_weights_to_stimulus_presentations(session,fit)
plt.plot(session.stimulus_presentations.reward_rate,'m',label='Flash Reward')
plt.plot(session.stimulus_presentations.bout_rate,'g',label='Flash Lick')
plt.plot(session.stimulus_presentations.running_rate/20,'r')
plt.gca().axhline(0,linestyle='--',alpha=0.5,color='k')
plt.gca().axhline(2/80,linestyle='--',alpha=0.5,color='m')
plt.gca().axhline(.1,linestyle='--',alpha=0.5,color='g')
plt.xlabel('Flash #',fontsize=12)
plt.ylabel('Rate/Flash',fontsize=12)
plt.legend()
plt.xlim([0,len(session.stimulus_presentations)])
plt.ylim([0,1])

plt.figure()
plt.plot(session.stimulus_presentations.reward_rate, session.stimulus_presentations.bout_rate,'ko',alpha=.1)
plt.ylim([0, 0.4])
plt.plot([0,2/80],[0.1,0.1],linestyle='--',color='r',alpha=0.5)
plt.axvline(2/80,linestyle='--',color='r',alpha=0.5)
plt.xlim(xmin=0)
plt.ylabel('lick rate/flash')
plt.xlabel('reward rate/flash')


