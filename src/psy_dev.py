import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
from importlib import reload
plt.ion()


# Basic Characterization
dir="/home/alex.piet/codebase/behavior/psy_fits_v4/"
ids = ps.get_active_ids()
fit = ps.plot_fit(ids[0],directory=dir)
ps.plot_session_summary(ids,savefig=True,directory = dir)

for id in ids:
    print(id)
    try:
        fit = ps.load_fit(id, directory=dir)
        ps.summarize_fit(fit,directory=dir, savefig=True)
    except:
        pass


# get PCA plots
dropouts, hits,false_alarms,misses,ids = ps.get_all_dropout(ps.get_session_ids(),directory=dir)
mice_dropouts, mice_good_ids = ps.get_mice_dropout(ps.get_mice_ids(),directory=dir)
fit = ps.load_fit(ps.get_session_ids()[1],directory=dir)
pca = ps.PCA_on_dropout(dropouts, labels=fit['labels'], mice_dropouts=mice_dropouts,mice_ids=mice_good_ids, hits=hits,false_alarms=false_alarms, misses=misses,directory=dir)

# PCA on weights
all_weights = ps.plot_session_summary_weights(ps.get_session_ids(),return_weights=True)
x = np.vstack(all_weights)
task = x[:,2]
timing = np.mean(x[:,3:],1)
dex = task-timing
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(x)
X = pca.transform(x)
plt.figure()
scat = plt.gca().scatter(X[X[:,0] < 50,0],X[X[:,0] < 50,1],c=-dex[X[:,0] < 50],cmap='plasma')
cbar = plt.gcf().colorbar(scat, ax = plt.gca())
cbar.ax.set_ylabel('Task Weight Index',fontsize=12)
plt.gca().set_xlabel('PC 1')
plt.gca().set_ylabel('PC 1')

# Get unified clusters
ps.build_all_clusters(ps.get_session_ids(), save_results=True)

# Compare fits
dir1 = "/home/alex.piet/codebase/behavior/psy_fits_v3_01/"
dir2 = "/home/alex.piet/codebase/behavior/psy_fits_v3_11/"
dir3 = "/home/alex.piet/codebase/behavior/psy_fits_v3/"
dir4 = "/home/alex.piet/codebase/behavior/psy_fits_v2/"
dir5 = "/home/alex.piet/codebase/behavior/psy_fits_v4/"

ids = ps.get_session_ids()
all_roc = ps.compare_all_fits(ids, [dir1,dir2,dir3,dir4,dir5])
all_roc = ps.compare_all_fits(ids, [dir4,dir5])


# Comparing Timing versions
dir = "/home/alex.piet/codebase/behavior/psy_fits_v5/"
ids = ps.get_session_ids()
rocs = ps.compare_timing_versions(ids, dir)

########### DEV #######################
# Getting behavior sessions
from allensdk.brain_observatory.behavior import behavior_project_cache as bpc
cache = bpc.InternalCacheFromLims()

sessions = cache.get_sessions()
osid = sessions.iloc[0]['ophys_session_id']
session = cache.get_session(osid)

d = sessions.iloc[0]['donor_id']
bsessions = cache.get_all_behavior_sessions(d, exclude_imaging_sessions=True)
bsid = bsessions.iloc[0]['behavior_session_id']
bsession = cache.get_behavior_only_session(bsid)


