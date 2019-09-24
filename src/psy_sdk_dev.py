import psy_tools as ps
import psy_sdk_tools as psd
import psy_timing_tools as pt
import matplotlib.pyplot as plt
from alex_utils import *
plt.ion()



# clustering + SDK
session_ids = ps.get_session_ids()

# For an individual session, add cluster labels to stimulus_presentations_tables
id = session_ids[15]
fit = ps.load_fit(id)
session = ps.get_data(id)
stim = psd.add_clusters_to_stimulus_presentations(session,fit)

# If you also want the model weights:
stim = psd.add_weights_to_stimulus_presentations(session,fit)

# If you want to know this sessions timing/task index:
pt.get_session_task_index(id)

# For an individual session, load clusters, and add to flash_response_df
directory = '/home/alex.piet/codebase/behavior/psy_fits_v2/'
id = session_ids[15]
fit = ps.load_fit(id)
fit = ps.plot_fit(id, cluster_labels=fit['all_clusters']['4'][1])
session = ps.get_data(id)
stage = session.metadata['stage']
cdf = psd.get_joint_table(fit,session,use_all_clusters=True)
psd.mean_response_by_cluster(cdf,'4',session=id,stage = stage,filename=directory+str(id)+"_all_cluster_4_mean_response")
psd.running_behavior_by_cluster(cdf,'4',session=id,stage = stage)
psd.latency_behavior_by_cluster(cdf,'4',session=id,stage = stage)
plt.figure(); plt.plot(fit['all_clusters']['4'][1])

# for many sessions is slow and takes a lot of memory
# Need to also parse by cell type
cache = ps.get_cache()
manifest = ps.get_manifest()
set_ids = ps.get_intersection([ps.get_slc_session_ids(),ps.get_active_ids(), ps.get_B_ids()])
cdfs = psd.build_multi_session_joint_table(set_ids,cache, manifest, use_all_clusters=True,slim_df=True)
directory = '/home/alex.piet/codebase/behavior/psy_fits_v2/'
psd.mean_response_by_cluster(cdfs,'4',filename=directory+"Slc_active_B_mean_response")
psd.running_behavior_by_cluster(cdfs,'4',filename=directory+"Slc_active_B_running")
psd.latency_behavior_by_cluster(cdfs,'4',filename=directory+"Slc_active_B_latency")


# Look at variance explained by clustering
var_expl = psd.get_var_expl_by_cell(cdf,'4','mean_response')
cdf['shuffled_response'] = cdf['mean_response'].sample(frac=1).values
var_expl_shuffle = psd.get_var_expl_by_cell(cdf,'4','shuffled_response')
plt.figure()
plt.plot(np.sort(var_expl),'r')
plt.plot(np.sort(var_expl_shuffle),'k')
plt.axhspan(-np.max(np.abs(var_expl_shuffle)), np.max(np.abs(var_expl_shuffle)),color='k',alpha=.25)
plt.xlim([0, len(var_expl)])






