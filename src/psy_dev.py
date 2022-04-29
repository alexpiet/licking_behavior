import psy_tools as ps
import psy_general_tools as pgt
import psy_timing_tools as pt
import psy_metrics_tools as pm
 
## Basic SDK
###########################################################################################
bsid = 914705301
training = pgt.get_training_manifest()

test = training.drop_duplicates(keep='first',subset=['session_type'])
test = test[test.session_type.str.startswith('TRAINING')]
test = test.sort_values(by=['session_type'])


## PCA
###########################################################################################
drop_dex,drop_var = ps.PCA_dropout(ids,pgt.get_mice_ids(),version)
weight_dex  = ps.PCA_weights(ids,pgt.get_mice_ids(),version)
ps.PCA_analysis(ids, pgt.get_mice_ids(),version)
# Make nice version with markersize=4

###########################################################################################
# Basically the same as model manifest, but has different names
strategy_index_df = ps.get_all_timing_index(ids,version)
ps.plot_model_index_summaries(strategy_index_df,version)

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

