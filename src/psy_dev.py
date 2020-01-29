import psy_tools as ps
import psy_general_tools as pgt
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

TODO
1. manifest has duplicate entries of stage 3 for a container???!?!?!?
2. Need a consistent list of sessions. 
3. Try relaxing session constraints so I can look at incomplete containers?
'''
    
## dev
###########################################################################################
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

## Basic Analysis
###########################################################################################
directory="/home/alex.piet/codebase/behavior/psy_fits_v9/"
ids = pgt.get_active_ids()

# Plot Example session
fit = ps.plot_fit(ids[0],directory=directory)

# Basic Characterization, Summaries of each session
ps.summarize_fits(ids,directory)

# Basic Characterization, Summaries at Session level
ps.plot_session_summary(ids,savefig=True,directory = directory)

## PCA
###########################################################################################
drop_dex    = ps.PCA_dropout(ids,pgt.get_mice_ids(),directory)
weight_dex  = ps.PCA_weights(ids,pgt.get_mice_ids(),directory)
ps.PCA_analysis(ids, pgt.get_mice_ids(),directory)

df = ps.get_all_timing_index(ids,directory)
ps.plot_model_index_summaries(df,directory)

## Build Table of Mice by Strategy, cre line and depth
###########################################################################################
model_manifest = ps.build_model_manifest(directory=directory,container_in_order=True)

ps.plot_all_manifest_by_stage(model_manifest, directory=directory)
ps.compare_all_manifest_by_stage(model_manifest, directory=directory)

ps.plot_all_manifest_by_stage(model_manifest.query('trained_A'), directory=directory,group_label='TrainedA')
ps.compare_all_manifest_by_stage(model_manifest.query('trained_A'), directory=directory,group_label='TrainedA')

ps.plot_all_manifest_by_stage(model_manifest.query('trained_B'), directory=directory,group_label='TrainedB')
ps.compare_all_manifest_by_stage(model_manifest.query('trained_B'), directory=directory,group_label='TrainedB')

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


