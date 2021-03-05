import psy_tools as ps
import psy_general_tools as pgt
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
from alex_utils import *
from importlib import reload
from visual_behavior.translator.allensdk_sessions import sdk_utils
plt.ion()

'''
Have to modify sdk code by:
SDK 1.6.0
1. Have to set up mtrain and LIMS passwords in ~/.bash_profile
2. No further need for @memoize decorator anymore

SDK 1.3.0:
1. adding mtrain password in two files:
  allensdk/internal/api/behavior_data_lims_api.py
  allensdk/brain_observatory/behavior/behavior_project_lims_api.py
2. @memoize decorator to: 
  get_licks(), get_rewards(), get_trials(), get_metadata(), get_stimulus_presentations()
  allensdk/internal/api/behavior_data_lims_api.py
3. Changes to codebase
    a. all inputs are bsids, with OPHYS, the switch to the relevant osid happens at the data interface level
    b. all mouse ids are donor_ids, not specimen_ids

'''
 
## Basic SDK
###########################################################################################
bsid = 914705301
manifest = pgt.get_ophys_manifest()
training = pgt.get_training_manifest()

## Basic Analysis
###########################################################################################
directory="/home/alex.piet/codebase/behavior/psy_fits_v20/"
ids = pgt.get_active_ids()

# Plot Example session
fit = ps.plot_fit(ids[0],version=20)

# Basic Characterization, Summaries of each session
ps.summarize_fits(ids,version=20)

# Basic Characterization, Summaries at Session level
ps.plot_session_summary(ids,savefig=True,version=20)

## PCA
###########################################################################################
drop_dex,drop_var = ps.PCA_dropout(ids,pgt.get_mice_ids(),version)
weight_dex  = ps.PCA_weights(ids,pgt.get_mice_ids(),version)
ps.PCA_analysis(ids, pgt.get_mice_ids(),version)

strategy_index_df = ps.get_all_timing_index(ids,version)
ps.plot_model_index_summaries(strategy_index_df,version)

## Build Table of Mice by Strategy, cre line and depth
###########################################################################################
model_manifest = ps.build_model_manifest(version,container_in_order=True)

# Main Analyses
ps.plot_all_manifest_by_stage(model_manifest, version)
ps.compare_all_manifest_by_stage(model_manifest, version)

# Cosyne figures
ps.plot_manifest_by_stage(model_manifest,'lick_hit_fraction',version=version,fs1=24,fs2=16,filetype='.svg')
ps.plot_manifest_by_stage(model_manifest,'lick_fraction',version=version,fs1=24,fs2=16,filetype='.svg')
ps.plot_manifest_by_stage(model_manifest,'trial_hit_fraction',version=version,fs1=24,fs2=16,filetype='.svg')
ps.plot_manifest_by_stage(model_manifest,'strategy_dropout_index',version=version,fs1=24,fs2=16,filetype='.svg')

# Additional Analyses I haven't organized yet
ps.plot_manifest_groupby(model_manifest, 'lick_hit_fraction','task_strategy_session',version)
ps.plot_manifest_groupby(model_manifest, 'num_hits','task_strategy_session',version)

ps.scatter_manifest(model_manifest, 'strategy_dropout_index','lick_hit_fraction', version)
ps.scatter_manifest(model_manifest, 'task_only_dropout_index','lick_hit_fraction', version,sflip1=True)
ps.scatter_manifest(model_manifest, 'timing_only_dropout_index','lick_hit_fraction', version,sflip1=True)
ps.scatter_manifest(model_manifest, 'task_only_dropout_index','timing_only_dropout_index', version,sflip1=True,sflip2=True,cindex='lick_hit_fraction')
ps.plot_manifest_by_date(model_manifest,version)
ps.plot_task_timing_over_session(model_manifest,version)
ps.plot_task_timing_by_training_duration(model_manifest,version)

## Look by Cre Line
ps.plot_all_manifest_by_cre(model_manifest, version)
ps.plot_task_index_by_cre(model_manifest,version)
ps.plot_manifest_by_cre(model_manifest,'lick_hit_fraction',version=version,savefig=True,group_label='all_',fs1=20,fs2=16,labels=['Slc','Sst','Vip'],figsize=(5,4),ylabel='Lick Hit Fraction')

## Look at Trained A Mice
ps.plot_all_manifest_by_stage(model_manifest.query('trained_A'), version=version,group_label='TrainedA')
ps.compare_all_manifest_by_stage(model_manifest.query('trained_A'), version=version,group_label='TrainedA')

## Look at Trained B Mice
ps.plot_all_manifest_by_stage(model_manifest.query('trained_B'), version=version,group_label='TrainedB')
ps.compare_all_manifest_by_stage(model_manifest.query('trained_B'), version=version,group_label='TrainedB')


###########################################################################################
###########################################################################################
#### DEVELOPMENT CODE BELOW HERE ####
###########################################################################################
###########################################################################################
   
## Clustering
###########################################################################################
# Get unified clusters
ps.build_all_clusters(pgt.get_active_ids(), save_results=True)

## Compare fits. These comparisons are not exact, because some fits crashed on each version
## v8 and earlier use ophys_experiment_ids, not behavior_session_ids as indicies 
########################################################################################### 
dir1 = "/home/alex.piet/codebase/behavior/psy_fits_v1/"
dir2 = "/home/alex.piet/codebase/behavior/psy_fits_v2/"
dir3a = "/home/alex.piet/codebase/behavior/psy_fits_v3_01/"
dir3b= "/home/alex.piet/codebase/behavior/psy_fits_v3_11/"
dir3 = "/home/alex.piet/codebase/behavior/psy_fits_v3/"
dir4 = "/home/alex.piet/codebase/behavior/psy_fits_v4/"
dir5 = "/home/alex.piet/codebase/behavior/psy_fits_v5/"
dir6 = "/home/alex.piet/codebase/behavior/psy_fits_v6/"
dir7 = "/home/alex.piet/codebase/behavior/psy_fits_v7/"
dir8 = "/home/alex.piet/codebase/behavior/psy_fits_v8/"

dirs = [dir1,dir2,dir3,dir3a,dir3b,dir4,dir5,dir6,dir7,dir8]
dirs = [dir6,dir7,dir8]

# Plot development history
all_roc = ps.compare_versions(dirs, ids)
ps.compare_versions_plot(all_roc)

# Comparing Timing versions
rocs = ps.compare_timing_versions(ids,"/home/alex.piet/codebase/behavior/psy_fits_v5/")

## Compare with Late-Task
########################################################################################### 
dir10 = "/home/alex.piet/codebase/behavior/psy_fits_v10/"
dir11 = "/home/alex.piet/codebase/behavior/psy_fits_v11/"
dirs = [dir10, dir11]
ids = pgt.get_active_ids()

all_roc = ps.compare_versions(dirs, ids)
ps.compare_versions_plot(all_roc)

ps.plot_session_summary(ids,savefig=True,directory = dir11,nel=4)

