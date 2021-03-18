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





###################
# Population Summary Figures
import seaborn as sns
sns.set_context('notebook', font_scale=1, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': False, 'ytick.left': False,})


lick_rates, reward_rates, all_epochs,times, count, all_times,all_hit_fraction,all_hit_rate,all_fa_rate,all_dprime,criterion,IDS_out, num_hits = pm.unpack_df(df)
pm.plot_all_epochs(all_epochs,'all')                  
pm.plot_all_times(times,count,all_times,'all')       
pm.plot_all_rates(lick_rates,reward_rates,'all')
pm.plot_all_dprime(all_dprime,criterion,'all')
pm.plot_all_performance_rates(all_hit_fraction,all_hit_rate,all_fa_rate,'all')
pm.plot_all_rates_averages(lick_rates,reward_rates,'all')
pm.plot_all_performance_rates_averages(all_dprime,criterion, all_hit_fraction,all_hit_rate,all_fa_rate,'all')
pm.compare_all_rates_averages([lick_rates],[reward_rates],rlabels=['All'],label='all',split_on=2400)
pm.compare_all_performance_rates_averages([all_dprime],[criterion],[all_hit_fraction],[all_hit_rate],[all_fa_rate],['All'],'all',split_on=2400)




# by cre line
############################################################
lick_rates_slc,reward_rates_slc,all_epochs_slc,times_slc,count_slc,all_times_slc,all_hit_fraction_slc,all_hit_rate_slc,all_fa_rate_slc,all_dprime_slc,criterion_slc,slc_ids, slc_hits=pm.query_get_rates(df,'cre_line=="Slc17a7-IRES2-Cre"')
lick_rates_vip, reward_rates_vip, all_epochs_vip,times_vip, count_vip, all_times_vip,all_hit_fraction_vip,all_hit_rate_vip,all_fa_rate_vip,all_dprime_vip,criterion_vip,vip_ids,vip_hits = pm.query_get_rates(df,'cre_line=="Vip-IRES-Cre"')
lick_rates_sst, reward_rates_sst, all_epochs_sst,times_sst, count_sst, all_times_sst,all_hit_fraction_sst,all_hit_rate_sst,all_fa_rate_sst,all_dprime_sst,criterion_sst,sst_ids,sst_hits = pm.query_get_rates(df,'cre_line=="Sst-IRES-Cre"')

# Rates over Time

pm.compare_all_rates([lick_rates_slc,lick_rates_vip,lick_rates_sst],[reward_rates_slc,reward_rates_vip,reward_rates_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_dprime([all_dprime_slc,all_dprime_vip,all_dprime_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_criterion([criterion_slc,criterion_vip,criterion_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_performance_rates([all_hit_fraction_slc,all_hit_fraction_vip,all_hit_fraction_sst],[all_hit_rate_slc,all_hit_rate_vip,all_hit_rate_sst],[all_fa_rate_slc,all_fa_rate_vip,all_fa_rate_sst],['SLC','VIP','SST'],'slc_vip_sst')

# Histograms

pm.compare_hit_count([slc_hits,vip_hits,sst_hits],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_rates_averages([lick_rates_slc,lick_rates_vip,lick_rates_sst],[reward_rates_slc,reward_rates_vip,reward_rates_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_performance_rates_averages([all_dprime_slc,all_dprime_vip,all_dprime_sst], [criterion_slc,criterion_vip,criterion_sst],[all_hit_fraction_slc,all_hit_fraction_vip,all_hit_fraction_sst],[all_hit_rate_slc,all_hit_rate_vip,all_hit_rate_sst],[all_fa_rate_slc,all_fa_rate_vip,all_fa_rate_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_performance_rates_averages([all_dprime_slc,all_dprime_vip,all_dprime_sst], [criterion_slc,criterion_vip,criterion_sst],[all_hit_fraction_slc,all_hit_fraction_vip,all_hit_fraction_sst],[all_hit_rate_slc,all_hit_rate_vip,all_hit_rate_sst],[all_fa_rate_slc,all_fa_rate_vip,all_fa_rate_sst],['SLC','VIP','SST'],'slc_vip_sst',split_on=2400)

# Clustering

pm.compare_all_times([times_slc,times_vip, times_sst],[count_slc, count_vip,count_sst],[all_times_slc,all_times_vip,all_times_sst],['SLC','VIP','SST'],'slc_vip_sst')
pm.compare_all_epochs([all_epochs_slc,all_epochs_vip, all_epochs_sst],['SLC','VIP','SST'],'slc_vip_sst',smoothing=500)


# A v B
############################################################
lick_ratesA, reward_ratesA, all_epochsA,timesA, countA, all_timesA, all_hfA, all_hrA, all_faA, all_dprimeA, criterionA, IDS_outA, num_hitsA = pm.query_get_rates(df,'image_set=="A"')
lick_ratesB, reward_ratesB, all_epochsB,timesB, countB, all_timesB, all_hfB, all_hrB, all_faB, all_dprimeB, criterionB, IDS_outB, num_hitsB = pm.query_get_rates(df,'image_set=="B"')

# Rates over Time

pm.compare_all_rates([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],'A_B')
pm.compare_all_dprime([all_dprimeA,all_dprimeB],['A','B'],'A_B')
pm.compare_all_criterion([criterionA,criterionB],['A','B'],'A_B')
pm.compare_all_performance_rates([all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],'A_B')

# histograms

pm.compare_hit_count([num_hitsA,num_hitsB],['A','B'],'A_B')
pm.compare_all_rates_averages([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],'A_B')
pm.compare_all_rates_averages([lick_ratesA,lick_ratesB],[reward_ratesA,reward_ratesB],['A','B'],'A_B',split_on=2400)
pm.compare_all_performance_rates_averages([all_dprimeA,all_dprimeB],[criterionA,criterionB],[all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],'A_B')
pm.compare_all_performance_rates_averages([all_dprimeA,all_dprimeB],[criterionA,criterionB],[all_hfA,all_hfB],[all_hrA,all_hrB],[all_faA,all_faB],['A','B'],'A_B',split_on=2400)
pm.compare_all_performance_rates_averages_dprime([all_dprimeA,all_dprimeB],['A','B'],'A_B')
pm.compare_all_performance_rates_averages_dprime([all_dprimeA,all_dprimeB],['A','B'],'A_B',split_on=2400)
pm.compare_all_performance_rates_averages_criterion([criterionA,criterionB],['A','B'],'A_B')
pm.compare_all_performance_rates_averages_criterion([criterionA,criterionB],['A','B'],'A_B',split_on=2400)

# Clustering

pm.compare_all_times([timesA,timesB],[countA,countB],[all_timesA,all_timesB],['A','B'],'A_B')
pm.compare_all_epochs([all_epochsA,all_epochsB],['A','B'],'A_B',smoothing=500)


# by stage
############################################################
lick_rates1, reward_rates1, all_epochs1,times1, count1, all_times1, all_hf1, all_hr1, all_fa1, all_dprime1, criterion1, IDS_out1, num_hits1 = pm.query_get_rates(df,'stage =="1"')
lick_rates3, reward_rates3, all_epochs3,times3, count3, all_times3, all_hf3, all_hr3, all_fa3, all_dprime3, criterion3, IDS_out3, num_hits3 = pm.query_get_rates(df,'stage =="3"')
lick_rates4, reward_rates4, all_epochs4,times4, count4, all_times4, all_hf4, all_hr4, all_fa4, all_dprime4, criterion4, IDS_out4, num_hits4 = pm.query_get_rates(df,'stage =="4"')
lick_rates6, reward_rates6, all_epochs6,times6, count6, all_times6, all_hf6, all_hr6, all_fa6, all_dprime6, criterion6, IDS_out6, num_hits6 = pm.query_get_rates(df,'stage =="6"')

# Rates over time

pm.compare_all_rates([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'],'by_stage')
pm.compare_all_dprime([all_dprime1,all_dprime3,all_dprime4,all_dprime6],['1','3','4','6'],'by_stage')
pm.compare_all_criterion([criterion1,criterion3,criterion4,criterion6],['1','3','4','6'],'by_stage')
pm.compare_all_performance_rates([all_hf1,all_hf3,all_hf4,all_hf6],[all_hr1,all_hr3,all_hr4,all_hr6],[all_fa1,all_fa3,all_fa4,all_fa6],['1','3','4','6'],'by_stage')

# Histograms

pm.compare_hit_count([num_hits1,num_hits3,num_hits4,num_hits6],['1','3','4','6'],'by_stage')
pm.compare_all_rates_averages([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'],'by_stage')
pm.compare_all_rates_averages([lick_rates1,lick_rates3,lick_rates4,lick_rates6],[reward_rates1,reward_rates3,reward_rates4,reward_rates6],['1','3','4','6'],'by_stage',split_on=2400)
pm.compare_all_performance_rates_averages([all_dprime1,all_dprime3,all_dprime4,all_dprime6],[criterion1,criterion3,criterion4,criterion6],[all_hf1,all_hf3,all_hf4,all_hf6],[all_hr1,all_hr3,all_hr4,all_hr6],[all_fa1,all_fa3,all_fa4,all_fa6],['1','3','4','6'],'by_stage')
pm.compare_all_performance_rates_averages([all_dprime1,all_dprime3,all_dprime4,all_dprime6],[criterion1,criterion3,criterion4,criterion6],[all_hf1,all_hf3,all_hf4,all_hf6],[all_hr1,all_hr3,all_hr4,all_hr6],[all_fa1,all_fa3,all_fa4,all_fa6],['1','3','4','6'],'by_stage',split_on=2400)

# Just 3 v 4

pm.compare_all_rates([lick_rates3,lick_rates4],[reward_rates3,reward_rates4],['3','4'],'3_4')
pm.compare_all_dprime([all_dprime3,all_dprime4],['3','4'],'3_4')
pm.compare_all_criterion([criterion3,criterion4],['3','4'],'3_4')
pm.compare_all_performance_rates([all_hf3,all_hf4],[all_hr3,all_hr4],[all_fa3,all_fa4],['3','4'],'3_4')

# Clustering

pm.compare_all_times([times1,times3,times4,times6],[count1,count3,count4,count6],[all_times1,all_times3,all_times4,all_times6],['1','3','4','6'],'by_stage')
pm.compare_all_epochs([all_epochs3,all_epochs4],['3','4'],'3_4',smoothing=500) 


# TrainedA vs TrainedB
############################################################
lick_ratesAt, reward_ratesAt, all_epochsAt,timesAt, countAt, all_timesAt, all_hfAt, all_hrAt, all_faAt, all_dprimeAt, criterionAt, IDS_outAt, num_hitsAt = pm.query_get_rates(df,'trained_A')
lick_ratesBt, reward_ratesBt, all_epochsBt,timesBt, countBt, all_timesBt, all_hfBt, all_hrBt, all_faBt, all_dprimeBt, criterionBt, IDS_outBt, num_hitsBt = pm.query_get_rates(df,'not trained_A')

# Rates over time

pm.compare_all_rates([lick_ratesAt,lick_ratesBt],[reward_ratesAt,reward_ratesBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_dprime([all_dprimeAt,all_dprimeBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_criterion([criterionAt,criterionBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_performance_rates([all_hfAt,all_hfBt],[all_hrAt,all_hrBt],[all_faAt,all_faBt],['At','Bt'],'TrainedA_TrainedB')

# Histograms

pm.compare_hit_count([num_hitsAt,num_hitsBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_rates_averages([lick_ratesAt,lick_ratesBt],[reward_ratesAt,reward_ratesBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_rates_averages([lick_ratesAt,lick_ratesBt],[reward_ratesAt,reward_ratesBt],['At','Bt'],'TrainedA_TrainedB',split_on=2400)
pm.compare_all_performance_rates_averages([all_dprimeAt,all_dprimeBt],[criterionAt,criterionBt],[all_hfAt,all_hfBt],[all_hrAt,all_hrBt],[all_faAt,all_faBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_performance_rates_averages([all_dprimeAt,all_dprimeBt],[criterionAt,criterionBt],[all_hfAt,all_hfBt],[all_hrAt,all_hrBt],[all_faAt,all_faBt],['At','Bt'],'TrainedA_TrainedB',split_on=2400)
pm.compare_all_performance_rates_averages_dprime([all_dprimeAt,all_dprimeBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_performance_rates_averages_dprime([all_dprimeAt,all_dprimeBt],['At','Bt'],'TrainedA_TrainedB',split_on=2400)
pm.compare_all_performance_rates_averages_criterion([criterionAt,criterionBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_performance_rates_averages_criterion([criterionAt,criterionBt],['At','Bt'],'TrainedA_TrainedB',split_on=2400)
pm.compare_all_performance_rates_averages_hit_fraction([all_hfAt,all_hfBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_performance_rates_averages_hit_fraction([all_hfAt,all_hfBt],['At','Bt'],'TrainedA_TrainedB',split_on=2400)
pm.compare_all_performance_rates_averages_hit_rate([all_hrAt,all_hrBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_performance_rates_averages_hit_rate([all_hrAt,all_hrBt],['At','Bt'],'TrainedA_TrainedB',split_on=2400)
pm.compare_all_performance_rates_averages_false_alarm([all_faAt,all_faBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_performance_rates_averages_false_alarm([all_faAt,all_faBt],['At','Bt'],'TrainedA_TrainedB',split_on=2400)


# Clustering

pm.compare_all_times([timesAt,timesBt],[countAt,countBt],[all_timesAt,all_timesBt],['At','Bt'],'TrainedA_TrainedB')
pm.compare_all_epochs([all_epochsAt,all_epochsBt],['At','Bt'],'TrainedA_TrainedB',smoothing=500)


