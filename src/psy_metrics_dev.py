import numpy as np
import psy_general_tools as pgt
import matplotlib.pyplot as plt
import psy_metrics_tools as pm
import pandas as pd
from scipy import stats
import seaborn as sns
plt.ion()

# Basic Example
ophys = pgt.get_ophys_manifest()
bsid = ophys['behavior_session_ids'].values[0]
session = pgt.get_data(bsid)    # Get SDK session object
pm.get_metrics(session)         # annotate session
pm.plot_metrics(session)        # plots metrics for this session
durations = pm.get_time_in_epochs(session) # Duration of each epoch

# Plot figure for each session
pm.plot_all_metrics(pgt.get_ophys_manifest())
pm.plot_all_metrics(pgt.get_training_manifest())

# Build summary df (VERY SLOW)
df = pm.build_metrics_df()
train_df = pm.build_metrics_df(TRAIN=True)

# get summary df
df = pm.get_metrics_df()
train_df = pm.get_metrics_df(TRAIN=True)

# Population Summary Figures
pm.plot_rates_summary(df)
pm.plot_rates_summary(df,group='cre_line')
pm.plot_rates_summary(df,group='session_type')

pm.plot_count_summary(df, group='cre_line')
pm.plot_count_summary(df, group='session_type')
pm.plot_counts(df, ['fraction_low_lick_low_reward','fraction_high_lick_high_reward','fraction_high_lick_low_reward'] ,ylim=(0,1),label='epoch')

### TODO
# improve color/linestyle definitions
# Remove old functions
# Add significance testing
# Add average rates
# add averages rates 1st/2nd half

###################
pm.plot_all_epochs(all_epochs,'all')                  
pm.plot_all_times(times,count,all_times,'all')       
pm.plot_all_rates(lick_rates,reward_rates,'all')
pm.plot_all_dprime(all_dprime,criterion,'all')
pm.plot_all_performance_rates(all_hit_fraction,all_hit_rate,all_fa_rate,'all')
pm.plot_all_rates_averages(lick_rates,reward_rates,'all')
pm.plot_all_performance_rates_averages(all_dprime,criterion, all_hit_fraction,all_hit_rate,all_fa_rate,'all')
pm.compare_all_rates_averages([lick_rates],[reward_rates],rlabels=['All'],label='all',split_on=2400)
pm.compare_all_performance_rates_averages([all_dprime],[criterion],[all_hit_fraction],[all_hit_rate],[all_fa_rate],['All'],'all',split_on=2400)

pm.compare_all_rates([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],'A_B')
pm.compare_all_dprime([all_dprimeA,all_dprimeB],['A','B'],'A_B')
pm.compare_all_criterion([criterionA,criterionB],['A','B'],'A_B')
pm.compare_all_performance_rates([all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],'A_B')

pm.compare_hit_count([num_hitsA,num_hitsB],['A','B'],'A_B')
pm.compare_all_rates_averages([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],'A_B')
pm.compare_all_rates_averages([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],'A_B',split_on=2400)
pm.compare_all_performance_rates_averages([all_dprimeA,all_dprimeB],[criterionA,criterionB],[all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],'A_B')
pm.compare_all_performance_rates_averages([all_dprimeA,all_dprimeB],[criterionA,criterionB],[all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],'A_B',split_on=2400)
pm.compare_all_performance_rates_averages_dprime([all_dprimeA,all_dprimeB],['A','B'],'A_B')
pm.compare_all_performance_rates_averages_dprime([all_dprimeA,all_dprimeB],['A','B'],'A_B',split_on=2400)
pm.compare_all_performance_rates_averages_criterion([criterionA,criterionB],['A','B'],'A_B')
pm.compare_all_performance_rates_averages_criterion([criterionA,criterionB],['A','B'],'A_B',split_on=2400)

pm.compare_all_times([timesA,timesB],[countA,countB],[all_timesA,all_timesB],['A','B'],'A_B')
pm.compare_all_epochs([all_epochsA,all_epochsB],['A','B'],'A_B',smoothing=500)


