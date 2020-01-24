import numpy as np
import psy_general_tools as pgt
import matplotlib.pyplot as plt
import psy_metrics_tools as pm
import pandas as pd
from scipy import stats
import seaborn as sns
plt.ion()

# Basic Example
id = pgt.get_session_ids()[-27]  # Pick Session 
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

if False:
    df,times,counts = pm.get_rates_df()
    df.to_csv(path_or_buf='/home/alex.piet/codebase/behavior/data/psy_metrics_df_all_01_24_2020.csv')
df = pd.read_csv(filepath_or_buf='/home/alex.piet/codebase/behavior/data/psy_metrics_df_all_01_24_2020.csv')

lick_rates, reward_rates, all_epochs,times, count, all_times,all_hit_fraction,all_hit_rate,all_fa_rate,all_dprime,IDS_out, num_hits = pm.unpack_df(df)
pm.plot_all_epochs(all_epochs,'all')                  
pm.plot_all_times(times,count,all_times,'all')       
pm.plot_all_rates(lick_rates,reward_rates,'all')
pm.plot_all_rates_averages(lick_rates,reward_rates,'all')
pm.compare_all_rates_averages([lick_rates],[reward_rates],rlabels=['All'],label='all',split_on=2400)
pm.plot_all_dprime(all_dprime,'all')
pm.plot_all_performance_rates(all_hit_fraction,all_hit_rate,all_fa_rate,'all')
pm.plot_all_performance_rates_averages(all_dprime,all_hit_fraction,all_hit_rate,all_fa_rate,'all')
pm.compare_all_performance_rates_averages([all_dprime],[all_hit_fraction],[all_hit_rate],[all_fa_rate],['All'],'all',split_on=2400)

# by cre line
lick_rates_slc,reward_rates_slc,all_epochs_slc,times_slc,count_slc,all_times_slc,all_hit_fraction_slc,all_hit_rate_slc,all_fa_rate_slc,all_dprime_slc,slc_ids, slc_hits=pm.query_get_rates(df,'cre_line=="Slc17a7-IRES2-Cre"')
lick_rates_vip, reward_rates_vip, all_epochs_vip,times_vip, count_vip, all_times_vip,all_hit_fraction_vip,all_hit_rate_vip,all_fa_rate_vip,all_dprime_vip,vip_ids,vip_hits = pm.query_get_rates(df,'cre_line=="Vip-IRES-Cre"')
lick_rates_sst, reward_rates_sst, all_epochs_sst,times_sst, count_sst, all_times_sst,all_hit_fraction_sst,all_hit_rate_sst,all_fa_rate_sst,all_dprime_sst,sst_ids,sst_hits = pm.query_get_rates(df,'cre_line=="Sst-IRES-Cre"') 
pm.compare_all_rates([lick_rates_slc,lick_rates_vip,lick_rates_sst],[reward_rates_slc,reward_rates_vip,reward_rates_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_rates_averages([lick_rates_slc,lick_rates_vip,lick_rates_sst],[reward_rates_slc,reward_rates_vip,reward_rates_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_dprime([all_dprime_slc,all_dprime_vip,all_dprime_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_performance_rates([all_hit_fraction_slc,all_hit_fraction_vip,all_hit_fraction_sst],[all_hit_rate_slc,all_hit_rate_vip,all_hit_rate_sst],[all_fa_rate_slc,all_fa_rate_vip,all_fa_rate_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_performance_rates_averages([all_dprime_slc,all_dprime_vip,all_dprime_sst],[all_hit_fraction_slc,all_hit_fraction_vip,all_hit_fraction_sst],[all_hit_rate_slc,all_hit_rate_vip,all_hit_rate_sst],[all_fa_rate_slc,all_fa_rate_vip,all_fa_rate_sst],['SLC','VIP','SST'],'slc_vip_sst')

# Population Summary Figures by stage
lick_ratesA, reward_ratesA, all_epochsA,timesA, countA, all_timesA, all_hfA, all_hrA, all_faA, all_dprimeA, IDS_outA, num_hitsA = pm.query_get_rates(df,'image_set=="A"')
lick_ratesB, reward_ratesB, all_epochsB,timesB, countB, all_timesB, all_hfB, all_hrB, all_faB, all_dprimeB, IDS_outB, num_hitsB = pm.query_get_rates(df,'image_set=="B"')
lick_rates1, reward_rates1, all_epochs1,times1, count1, all_times1, all_hf1, all_hr1, all_fa1, all_dprime1, IDS_out1, num_hits1 = pm.query_get_rates(df,'stage =="1"')
lick_rates3, reward_rates3, all_epochs3,times3, count3, all_times3, all_hf3, all_hr3, all_fa3, all_dprime3, IDS_out3, num_hits3 = pm.query_get_rates(df,'stage =="3"')
lick_rates4, reward_rates4, all_epochs4,times4, count4, all_times4, all_hf4, all_hr4, all_fa4, all_dprime4, IDS_out4, num_hits4 = pm.query_get_rates(df,'stage =="4"')
lick_rates6, reward_rates6, all_epochs6,times6, count6, all_times6, all_hf6, all_hr6, all_fa6, all_dprime6, IDS_out6, num_hits6 = pm.query_get_rates(df,'stage =="6"')

pm.compare_hit_count(num_hitsA,num_hitsB,'A_B')
pm.compare_all_performance_rates_averages_dprime([all_dprimeA,all_dprimeB],['A','B'],'A_B')
pm.compare_all_performance_rates_averages_dprime([all_dprimeA,all_dprimeB],['A','B'],'A_B',split_on=2400)

# A v B
pm.compare_all_rates([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],'A_B')
pm.compare_all_rates_averages([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],'A_B')
pm.compare_all_rates_averages([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],'A_B',split_on=2400)
pm.compare_all_dprime([all_dprimeA,all_dprimeB],['A','B'],'A_B')
pm.compare_all_performance_rates([all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],'A_B')
pm.compare_all_performance_rates_averages([all_dprimeA,all_dprimeB],[all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],'A_B')
pm.compare_all_performance_rates_averages([all_dprimeA,all_dprimeB],[all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],'A_B',split_on=2400)

# by stage
pm.compare_all_rates([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'],'by_stage')
pm.compare_all_rates_averages([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'],'by_stage')
pm.compare_all_rates_averages([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'],'by_stage',split_on=2400)
pm.compare_all_dprime([all_dprime1,all_dprime3,all_dprime4,all_dprime6],['1','3','4','6'],'by_stage')
pm.compare_all_performance_rates([all_hf1,all_hf3,all_hf4,all_hf6],[all_hr1,all_hr3,all_hr4,all_hr6],[all_fa1,all_fa3,all_fa4,all_fa6],['1','3','4','6'],'by_stage')
pm.compare_all_performance_rates_averages([all_dprime1,all_dprime3,all_dprime4,all_dprime6],[all_hf1,all_hf3,all_hf4,all_hf6],[all_hr1,all_hr3,all_hr4,all_hr6],[all_fa1,all_fa3,all_fa4,all_fa6],['1','3','4','6'],'by_stage')
pm.compare_all_performance_rates_averages([all_dprime1,all_dprime3,all_dprime4,all_dprime6],[all_hf1,all_hf3,all_hf4,all_hf6],[all_hr1,all_hr3,all_hr4,all_hr6],[all_fa1,all_fa3,all_fa4,all_fa6],['1','3','4','6'],'by_stage',split_on=2400)

# Just 3 v 4
pm.compare_all_rates([lick_rates3,lick_rates4],[reward_rates3,reward_rates4],['3','4'],'3_4')
pm.compare_all_dprime([all_dprime3,all_dprime4],['3','4'],'3_4')
pm.compare_all_performance_rates([all_hf3,all_hf4],[all_hr3,all_hr4],[all_fa3,all_fa4],['3','4'],'3_4')

# Clustering
pm.compare_all_times([timesA,timesB],[countA,countB],[all_timesA,all_timesB],['A','B'],'A_B')
pm.compare_all_times([times1,times3,times4,times6],[count1,count3,count4,count6],[all_times1,all_times3,all_times4,all_times6],['1','3','4','6'],'by_stage')
pm.compare_all_epochs([all_epochsA,all_epochsB],['A','B'],'A_B',smoothing=500)
pm.compare_all_epochs([all_epochs3,all_epochs4],['3','4'],'3_4',smoothing=500) 

