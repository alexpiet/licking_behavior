import psy_tools as ps
import psy_analysis as pa
import psy_output_tools as po
import psy_general_tools as pgt
import psy_visualization as pv
import build_timing_regressor as pb
import matplotlib.pyplot as plt
plt.ion()
from importlib import reload
from alex_utils.alex_utils import *


# Quick start
################################################################################
version=21
summary_df  = po.get_ophys_summary_table(version)
change_df = po.get_change_table(version)
licks_df = po.get_licks_table(version)
bouts_df = po.build_bout_table(licks_df)


################################################################################
# Look at a single session for a single version
bsid = pgt.get_debugging_id(1) 
session = pgt.get_data(bsid)

# load fit for a single session
fit = ps.load_fit(bsid, version)
fit = ps.plot_fit(bsid,version=version)

# Build strategy df files for a session. This is done when the
# model is fit, but if you want to do it manually. 
ps.build_session_strategy_df(bsid, version) 
session_df = ps.load_session_strategy_df(bsid, version)
session_licks_df = ps.load_session_licks_df(bsid,version) 
session_bouts_df = po.build_bout_table(session_licks_df) 


## Versions
################################################################################
# Make a new version
version = 22
po.make_version(version)

# Get directory for a version
directory = pgt.get_directory(version) # main directory
figs_dir  = pgt.get_directory(version, subdirectory='figures')
fits_dir  = pgt.get_directory(version, subdirectory='fits')
stdf_dir  = pgt.get_directory(version, subdirectory='strategy_df')

# See what model versions are available
versions = po.get_model_versions(vrange=[20,25])

# Build inventory table
inventory_table = po.build_inventory_table(vrange=[20,25])
inventory = po.get_model_inventory(version)

# Build summary tables 
summary_df = po.build_summary_table(version)
change_df, crashed= po.build_change_table(summary_df, version) 
licks_df, crashed = po.build_licks_table(summary_df, version)

# Compare across versions
merged_df = po.build_comparison_df(summary_df_20, summary_df_21,'20','21')
pv.compare_across_versions(merged_df,'session_roc'],[20,21])


## Useful functions
################################################################################
strategies = pgt.get_strategy_list(version)
strings = pgt.get_clean_string(strings)


## Task Characterization 
################################################################################

# Build table of all changes
change_df = po.get_change_table(version)

# Plot the number of times each image pair is repeated per session
pv.plot_image_pair_repetitions(change_df, version)

# plot the number of image changes per session
pv.histogram_df(summary_df, 'num_changes',version=version)

# plot the number of image repeats between changes
pv.plot_image_repeats(change_df, version)


## Model Free Analysis
################################################################################

# Build table of all licks
licks_df = po.get_licks_table(version)

# Build table of licking bouts
bouts_df = po.build_bout_table(licks_df)

# Plot a histogram of inter-lick-intervals
pv.plot_interlick_interval(licks_df,version=version)
pv.plot_interlick_interval(licks_df,version=version,categories='rewarded')

# Plot duration of bouts in seconds and licks
pv.plot_bout_durations(bouts_df,version)

# Plot a histogram of inter-bout-intervals
pv.plot_interlick_interval(bouts_df,key='pre_ibi',version=version)
pv.plot_interlick_interval(bouts_df,key='pre_ibi',version=version,
    categories='bout_rewarded')

# Plot histogram of inter-bout-intervals following hits and misses
pv.plot_interlick_interval(bouts_df,key='pre_ibi',version=version,
    categories='post_reward')
pv.plot_interlick_interval(bouts_df,key='pre_ibi_from_start',version=version,
    categories='post_reward')

# Plot chronometric plot of hit %
pv.plot_chronometric(bouts_df, version)


## Analysis
################################################################################

# Visualize session
session = pgt.get_data(bsid)
pv.plot_session(session)
pv.plot_session(session,detailed=True)
pv.plot_session(session,detailed=True,fit=fit)
pv.plot_session_metrics(session)

# Load summary tables
version =21
summary_df  = po.get_ophys_summary_table(version)
change_df   = po.get_change_table(version)
licks_df    = po.get_licks_table(version)
bouts_df    = po.build_bout_table(licks_df)

# Many plots
# This makes all the summary figures
pv.plot_session_summary(summary_df,version=version)

# plot strategy differences by cre-line
pv.plot_strategy_by_cre(summary_df,version)

# Makes plots of average value after splitting by groupby
pv.plot_all_df_by_experience(summary_df, version)
pv.plot_all_df_by_cre(summary_df, version)
pv.plot_all_pivoted_df_by_experience(summary_df,version)

# Individual plots
# Scatter two session wise metrics
pv.scatter_df(summary_df, 'strategy_dropout_index','lick_hit_fraction', version=version)
pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index', 
    version=version,flip1=True,flip2=True,cindex='lick_hit_fraction')

# Scatter a metric comparing across two matched sessions
pv.scatter_df_by_experience(summary_df,['Familiar','Novel 1'],
    'session_roc',experience_type='experience_level',version=version)
pv.scatter_df_by_experience(summary_df,['3','4'], 'session_roc',
    experience_type='session_number',version=version)
pv.histogram_df_by_experience(summary_df,['Familiar','Novel 1'],
    'session_roc',experience_type='experience_level',version=version)

# Plot average value of key after splitting by groupby 
pv.plot_df_groupby(summary_df,'num_hits','cre_line',version=version)
pv.plot_df_groupby(summary_df,'lick_hit_fraction','cre_line',version=version)

# plot average value of key relative to tracked sessions
pv.plot_pivoted_df_by_experience(summary_df,'strategy_dropout_index',version)

# Plot histogram of a metric either split by categorical groups or for entire summary_df
pv.histogram_df(summary_df, 'strategy_dropout_index',version=version)
pv.histogram_df(summary_df, 'strategy_dropout_index','cre_line',version=version)

# Plot values of metric by date collected
pv.plot_df_by_date(summary_df,'strategy_dropout_index',version)

# Look at trajectories over time
keys = ['RT','engaged','reward_rate','lick_hit_fraction_rate',
        'strategy_weight_index_by_image','lick_bout_rate','image_false_alarm']
for key in keys:
    pv.plot_session_summary_trajectory(summary_df,key,version,
        categories='visual_strategy_session')
    pv.plot_session_summary_trajectory(summary_df,key,version,
        categories='experience_level')


## Engagement
################################################################################

# Plot Engagement Landscape for all sessions
pv.plot_engagement_landscape(summary_df,version)

# Plot engagement for a single session
pv.plot_session_engagement(session, version)

# Plot Analysis of Engagement
pv.plot_engagement_analysis(summary_df,version)

# Look at engagement over time
keys = ['engaged']
for key in keys:
    pv.plot_session_summary_trajectory(summary_df,key,version,
        categories='visual_strategy_session')
    pv.plot_session_summary_trajectory(summary_df,key,version,
        categories='experience_level')

# Look at engagement compared to other rate metrics
pv.plot_engagement_comparison(summary_df,version)


## Response Times (RT)
################################################################################

# Plot RT split by engagement
pv.RT_by_engagement(summary_df,version)
pv.RT_by_engagement(summary_df.query('visual_strategy_session'),version,group='visual')
pv.RT_by_engagement(summary_df.query('not visual_strategy_session'),
    version,group='timing')

# Plot RT split by group
pv.RT_by_group(summary_df,version,engaged='engaged')
pv.RT_by_group(summary_df,version,engaged='disengaged')
pv.RT_by_group(summary_df,version,change_only=True)


## PCA 
################################################################################
pa.compute_PCA(summary_df,version,on='dropout')
pa.compare_PCA(summary_df,version)
pv.scatter_df_by_mouse(summary_df,'num_hits',ckey='strategy_dropout_index',
    version=version)


## Building the timing regressor
################################################################################
pb.build_timing_regressor()


## Splitting sessions by strategies
################################################################################
pv.view_strategy_labels(summary_df)

