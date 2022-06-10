import psy_tools as ps
import psy_output_tools as po
import psy_general_tools as pgt
import psy_visualization as pv
import matplotlib.pyplot as plt
plt.ion()
from importlib import reload
from alex_utils import *

# Quick start
################################################################################
version=21
summary_df  = po.get_ophys_summary_table(version)
change_df = po.get_change_table(version)
licks_df = po.get_licks_table(version)
bouts_df = po.build_bout_table(licks_df)

################################################################################
# Look at a single session for a single version
version = 21
bsid = pgt.get_debugging_id(1) 
session = pgt.get_data(bsid)

# load fit for a single session
fit = ps.load_fit(bsid, version)
fit = ps.plot_fit(bsid,version=version)

# Build strategy df files for a session. This is done when the
# model is fit, but if you want to do it manually. 
ps.build_session_strategy_df(bsid, version) 
ps.build_session_strategy_df(bsid, version, TRAIN=True)
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
#po.build_training_summary_table(version)# TODO Broken, Issue #92
#po.build_mouse_summary_table(version)   # TODO Broken, Issue #92

po.build_change_table(summary_df, version) # broken
licks_df, crashed = po.build_licks_table(summary_df, version)

## Useful functions
################################################################################
strategies = pgt.get_strategy_list(version)
strings = pgt.get_clean_string(strings)

## Task Characterization 
################################################################################

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
pv.plot_session_metrics(session)

# Load summary tables
version =21
summary_df  = po.get_ophys_summary_table(version)
training_df = po.get_training_summary_table(version) # TODO Broken, Issue #92
mouse_df    = po.get_mouse_summary_table(version)    # TODO Broken, Issue #92

# Many plots
# This makes all the summary figures
pv.plot_session_summary(summary_df,version=version)

# plot strategy differences by cre-line
pv.plot_strategy_by_cre(summary_df,version)

# Makes plots of average value after splitting by groupby
pv.plot_all_df_by_session_number(summary_df, version)
pv.plot_all_df_by_cre(summary_df, version)

# Individual plots
# Scatter two session wise metrics
pv.scatter_df(summary_df, 'strategy_dropout_index','lick_hit_fraction', version)
pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index', 
    version,flip1=True,flip2=True,cindex='lick_hit_fraction')

# Scatter a metric comparing across two matched sessions
pv.scatter_df_by_experience(summary_df,['3','4'], 'strategy_weight_index',version=version)
pv.scatter_df_by_experience(summary_df,['3','4'], 'session_roc',version=version)

# Plot average value of key after splitting by groupby 
pv.plot_df_groupby(summary_df,'num_hits','cre_line',version=version)
pv.plot_df_groupby(summary_df,'lick_hit_fraction','cre_line',version=version)

# Plot histogram of a metric either split by categorical groups or for entire summary_df
pv.histogram_df(summary_df, 'strategy_dropout_index',version)
pv.histogram_df(summary_df, 'strategy_dropout_index','cre_line',version)

# Plot values of metric by date collected
pv.plot_df_by_date(summary_df,'strategy_dropout_index',version)

## Engagement
################################################################################

# Plot Engagement Landscape for all sessions
pv.plot_engagement_landscape(summary_df,version)

# Plot engagement for a single session
pv.plot_session_engagement(session, version)

# Plot Analysis of Engagement
pv.plot_engagement_analysis(summary_df,version)

## Response Times (RT)
################################################################################

# Plot RT split by engagement
pv.RT_by_engagement(summary_df,version)
pv.RT_by_engagement(summary_df.query('visual_strategy_session'),version,group='visual')
pv.RT_by_engagement(summary_df.query('not visual_strategy_session'),
    version,group='timing')

# Plot RT split by group
pv.RT_by_group(summary_df,version)
pv.RT_by_group(summary_df,version,engaged=False)
pv.RT_by_group(summary_df,version,change_only=True)

## PCA # TODO, Issue #190
################################################################################
drop_dex,drop_var = ps.PCA_dropout(ids,pgt.get_mice_ids(),version)
weight_dex  = ps.PCA_weights(ids,pgt.get_mice_ids(),version)
ps.PCA_analysis(ids, pgt.get_mice_ids(),version)

## Event Triggered Analysis #TODO, Issue #225
################################################################################
pa.triggered_analysis(summary_df,version)




