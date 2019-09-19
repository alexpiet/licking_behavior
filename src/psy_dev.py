import psy_tools as ps
import psy_timing_tools as pt
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
plt.ion()

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

# Move everything from experiment_id to session_id?
# load_mouse
# load_session
# process_mouse
# get_all_mice is outdated, put in an error pointing to get_mice_ids
# same with get_all_ophys_IDS
# get_session_ids
# get_mice_ids
# get_mice_sessions





# get PCA plots
dropouts, hits,false_alarms,misses = ps.get_all_dropout(ps.get_session_ids())
mice_dropouts, mice_good_ids = ps.get_mice_dropout(ps.get_mice_ids())
fit = ps.load_fit(ps.get_session_ids()[0])
pca = ps.PCA_on_dropout(dropouts, labels=fit['labels'], mice_dropouts=mice_dropouts,mice_ids=mice_good_ids, hits=hits,false_alarms=false_alarms, misses=misses)

# Get unified clusters
ps.build_all_clusters(ps.get_session_ids(), save_results=True)






### Dev below here
ids = ps.get_session_ids()
directory = "/home/alex.piet/codebase/behavior/psy_fits/"
directory = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/psy_fits_v2/'
w,w_ids = ps.get_all_fit_weights(ids,directory=directory)
w_all = ps.merge_weights(w)
train,test = pc.split_data(w_all)
pc.plot_data(w_all)

ps.plot_session_summary(IDS)
ps.plot_session_summary(IDS,savefig=True)

# Start of filtering my session type
stages = ps.get_stage_names(IDS) # Takes Forever
ps.plot_session_summary(stages[1]+stages[3],savefig=True,group_label="A_")
ps.plot_session_summary(stages[4],savefig=True,group_label="B1_")
ps.plot_session_summary(stages[6],savefig=True,group_label="B2_")
ps.plot_session_summary(stages[4]+stages[6],savefig=True,group_label="B_")

good_IDS = ps.get_good_behavior_IDS(IDS) 
ps.plot_session_summary(good_IDS,savefig=True,group_label="hits_100_")

r = np.zeros((25,25))
for i in np.arange(1,25,1):
    for j in np.arange(1,25,1):
        r[i,j] = ps.compute_model_prediction_correlation(fit, fit_mov=i,data_mov=j)

import pandas as pd
behavior_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_sessions_to_load.h5', key='df')
all_flash_df = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_all_flash_df.h5', key='df')
behavior_psydata = ps.format_all_sessions(all_flash_df)
hyp2, evd2, wMode2, hess2, credibleInt2,weights2 = ps.fit_weights(behavior_psydata,TIMING4=True)
ypred2 = ps.compute_ypred(behavior_psydata, wMode2,weights2)
ps.plot_weights(wMode2, weights2,behavior_psydata,errorbar=credibleInt2, ypred = ypred2,validation=False,session_labels = behavior_sessions.stage_name.values)











