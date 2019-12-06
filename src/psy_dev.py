import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
from importlib import reload
plt.ion()


dir="/home/alex.piet/codebase/behavior/psy_fits_v6/"
ids = ps.get_active_ids()

# Basic Characterization, Summaries at Session level
ps.plot_session_summary(ids,savefig=True,directory = dir)

# Basic Characterization, Summaries of each session
ps.summarize_fits(ids,dir)

## PCA
###########################################################################################

# get PCA plots
ps.PCA_dropout(ids,ps.get_mice_ids(),dir)

# PCA on weights
ps.PCA_weights(ids,dir)

## Clustering
###########################################################################################
# Get unified clusters
ps.build_all_clusters(ps.get_active_ids(), save_results=True)

## Compare fits. These comparisons are not exact, because some fits crashed on each version
########################################################################################### 
dir0 = "/home/alex.piet/codebase/behavior/psy_fits_v1/"
dir1 = "/home/alex.piet/codebase/behavior/psy_fits_v2/"
dir2 = "/home/alex.piet/codebase/behavior/psy_fits_v3_01/"
dir3 = "/home/alex.piet/codebase/behavior/psy_fits_v3_11/"
dir4 = "/home/alex.piet/codebase/behavior/psy_fits_v3/"
dir5 = "/home/alex.piet/codebase/behavior/psy_fits_v4/"
dir6 = "/home/alex.piet/codebase/behavior/psy_fits_v5/"
dir7 = "/home/alex.piet/codebase/behavior/psy_fits_v6/"
dirs = [dir0, dir1,dir2,dir3,dir4,dir5,dir6,dir7]

# Plot development history
all_roc = ps.compare_versions(dirs, ids)
ps.compare_versions_plot(all_roc)

# Comparing Timing versions
rocs = ps.compare_timing_versions(ids,"/home/alex.piet/codebase/behavior/psy_fits_v5/")



