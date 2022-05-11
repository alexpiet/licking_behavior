import psy_tools as ps
import psy_output_tools as po
import psy_general_tools as pgt
import psy_visualization as pv
import matplotlib.pyplot as plt
plt.ion()
from importlib import reload
from alex_utils import *

################################################################################
# Look at a single session for a single version
version = 20
bsid = pgt.get_debugging_id(1) 
session = pgt.get_data(bsid)

# load fit for a single session
fit = ps.load_fit(bsid, version)
strategy_df = ps.load_session_strategy_df(bsid, version)
fit = ps.plot_fit(bsid,version=version)

# Build strategy df files for a session. This is done when the
# model is fit, but if you want to do it manually. 
ps.build_session_strategy_df(bsid, version) 
ps.build_session_strategy_df(bsid, version, TRAIN=True)
session_df = ps.load_session_strategy_df(bsid, version)

## Versions
################################################################################
# Make a new version
version = 21
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

# Build summary tables 
po.build_summary_table(version)
po.build_training_summary_table(version)# TODO Broken, Issue #92
po.build_mouse_summary_table(version)   # TODO Broken, Issue #92

## Useful functions
################################################################################
strategies = pgt.get_strategy_list(version)
strings = pgt.get_clean_string(strings)

## Analysis
################################################################################

# Load summary tables
version =20
summary_df = po.get_ophys_summary_table(version)
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
pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index', version,flip1=True,flip2=True,cindex='lick_hit_fraction')

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

# Plot Engagement Landscape
pv.plot_engagement_landscape(summary_df,version)

## PCA # TODO, Issue #190
###########################################################################################
drop_dex,drop_var = ps.PCA_dropout(ids,pgt.get_mice_ids(),version)
weight_dex  = ps.PCA_weights(ids,pgt.get_mice_ids(),version)
ps.PCA_analysis(ids, pgt.get_mice_ids(),version)

 




