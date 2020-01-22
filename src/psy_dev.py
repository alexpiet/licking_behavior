import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
from importlib import reload
from visual_behavior.translator.allensdk_sessions import sdk_utils
plt.ion()

'''
Have to modify sdk code by:
1. adding mtrain password in two files:
  allensdk/internal/api/behavior_data_lims_api.py
  allensdk/brain_observatory/behavior/behavior_project_lims_api.py
2. @memoize decorator to: 
  get_licks(), get_rewards(), get_trials(), get_metadata(), get_stimulus_presentations()
  allensdk/internal/api/behavior_data_lims_api.py

Changes to codebase
1. all inputs are bsids, with OPHYS, the switch to the relevant osid happens at the data interface level
2. all mouse ids are donor_ids, not specimen_ids
'''

# dev
oeid = 856096766
osid = sdk_utils.get_osid_from_oeid(oeid,pgt.get_cache())
bsid = sdk_utils.get_bsid_from_oeid(oeid,pgt.get_cache())

cache = pgt.get_cache()
ophys_sessions = cache.get_session_table()
ophys_experiments = cache.get_experiment_table()
behavior_sessions = cache.get_behavior_session_table()

session = pgt.get_data(bsid)
pm.annotate_licks(session)
pm.annotate_bouts(session)
ps.annotate_stimulus_presentations(session)
ps.process_session(bsid)

mouse_id = 834823464

#################
directory="/home/alex.piet/codebase/behavior/psy_fits_v8/"
ids = pgt.get_active_ids()

# Basic Characterization, Summaries at Session level
ps.plot_session_summary(ids,savefig=True,directory = directory)

# Basic Characterization, Summaries of each session
ps.summarize_fits(ids,directory)

ps.plot_fit(ids[0],directory=directory)

## PCA
###########################################################################################
drop_dex    = ps.PCA_dropout(ids,pgt.get_mice_ids(),directory)
weight_dex  = ps.PCA_weights(ids,pgt.get_mice_ids(),directory)
ps.PCA_analysis(ids, pgt.get_mice_ids(),directory)

df = ps.get_all_timing_index(ids,directory)
ps.plot_model_index_summaries(df,directory)

## Clustering
###########################################################################################
# Get unified clusters
ps.build_all_clusters(pgt.get_active_ids(), save_results=True)

## Compare fits. These comparisons are not exact, because some fits crashed on each version
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

## Build Table of Mice by Strategy, cre line and depth
###########################################################################################
model_manifest = ps.build_model_manifest(directory=directory,container_in_order=True)

ps.plot_manifest_by_stage(model_manifest,'session_roc',hline=0.5,ylims=[0.5,1],directory=directory)
ps.plot_manifest_by_stage(model_manifest,'lick_fraction',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'lick_hit_fraction',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'trial_hit_fraction',directory=directory)

ps.plot_manifest_by_stage(model_manifest,'task_dropout_index',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'task_weight_index',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'prior_bias',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'prior_task0',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'prior_omissions1',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'prior_timing1D',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_bias',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_task0',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_omissions1',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_timing1D',directory=directory)

# Looks like a real effect
ps.plot_manifest_by_stage(model_manifest,'task_dropout_index',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'task_weight_index',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_task0',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_task0_1st',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_task0_2nd',directory=directory)

ps.plot_manifest_by_stage(model_manifest,'avg_weight_timing1D',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_timing1D_1st',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_timing1D_2nd',directory=directory)

ps.plot_manifest_by_stage(model_manifest,'avg_weight_bias',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_bias_1st',directory=directory)
ps.plot_manifest_by_stage(model_manifest,'avg_weight_bias_2nd',directory=directory)

# Holds on a mouse by mouse basis as well
ps.compare_manifest_by_stage(model_manifest,['3','4'], 'task_weight_index',directory=directory)
ps.compare_manifest_by_stage(model_manifest,['3','4'], 'task_dropout_index',directory=directory)

ps.compare_manifest_by_stage(model_manifest,['3','4'], 'avg_weight_task0',directory=directory)
ps.compare_manifest_by_stage(model_manifest,['3','4'], 'avg_weight_timing1D',directory=directory)



