import psy_output_tools as po
import psy_general_tools as pgt

## Single Session
################################################################################
# Look at a single session for a single version
bsid = 951520319
session = pgt.get_data(bsid)

# load fit for a single session
fit = ps.load_fit(bsid, version)
strategy_df = ps.load_session_strategy_df(bsid, version)
fit = ps.plot_fit(bsid,version=version)

# Build strategy df files for a session. This is done when the
# model is fit, but if you want to do it manually. 
ps.build_session_strategy_df(bsid, version) 
ps.build_session_strategy_df(bsid, version, TRAIN=True)
strategy_fit = ps.load_session_strategy_df(bsid, version):

## Versions
################################################################################
# Make a new version
version = '21'
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

# Load summary tables
ophys_df    = po.get_ophys_summary_table(version)
training_df = po.get_training_summary_table(version) # TODO Broken, Issue #92
mouse_df    = po.get_mouse_summary_table(version)    # TODO Broken, Issue #92

## Useful functions
################################################################################
strategies = pgt.get_strategy_list(version)
strings = pgt.get_clean_string(strings)

## Analysis
################################################################################
summary_df = po.get_ophys_summary_df(version)

# This makes all the summary figures
pv.plot_session_summary(summary_df,version=version)

# Scatter two session wise metrics
# TODO, can we consolidate these? 
pv.scatter_df(summary_df, 'strategy_dropout_index','lick_hit_fraction', version)
pv.scatter_df(summary_df, 'visual_only_dropout_index','lick_hit_fraction', version,flip1=True)
pv.scatter_df(summary_df, 'timing_only_dropout_index','lick_hit_fraction', version,flip1=True)
pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index', version,flip1=True,flip2=True,cindex='lick_hit_fraction')

# Plot average value of key after splitting by groupby 
# TODO, What else do we want to plot?
pv.plot_df_groupby(summary_df, 'num_hits','cre_line',version=version)
ps.plot_df_groupby(model_manifest,'lick_hit_fraction','cre_line',version=version)
ps.plot_df_groupby(model_manifest,'strategy_dropout_index','cre_line',version=version,group='strategy_matched')
pv.plot_df_groupby(model_manifest,'lick_hit_fraction','session_number',version=version)
pv.plot_df_groupby(model_manifest,'lick_fraction','session_number',version=version)
pv.plot_df_groupby(model_manifest,'trial_hit_fraction','session_number',version=version)
pv.plot_df_groupby(model_manifest,'strategy_dropout_index','session_number',version=version)
  
# plots many things by session number
# TODO, can we consolidate?
pv.plot_all_df_by_session_number(summary_df, version)
pv.plot_all_df_by_cre(summary_df, version)

## Look at Trained A/B Mice
pv.plot_all_df_by_session_number(model_manifest.query('trained_A'), version=version,group='TrainedA')
pv.plot_all_df_by_session_number(model_manifest.query('trained_B'), version=version,group='TrainedB')

#TODO, compare manifest by stage? 
ps.compare_all_manifest_by_stage(model_manifest.query('trained_A'), version=version,group='TrainedA')
ps.compare_all_manifest_by_stage(model_manifest.query('trained_B'), version=version,group='TrainedB')


