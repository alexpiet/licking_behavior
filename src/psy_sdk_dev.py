import psy_tools as ps
import psy_sdk_tools as psd
import matplotlib.pyplot as plt
from alex_utils import *
plt.ion()

# clustering + SDK
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc
import allensdk.brain_observatory.behavior.swdb.utilities as tools

cache_json = {'manifest_path': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/visual_behavior_data_manifest.csv',
              'nwb_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/nwb_files',
              'analysis_files_base_dir': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/analysis_files',
              'analysis_files_metadata_path': '/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/SWDB_2019/analysis_files_metadata.json'
              }
cache = bpc.BehaviorProjectCache(cache_json)
manifest = cache.manifest
ids = manifest.ophys_experiment_id.values

# For an individual session
index = -7
id = ids[index]
stage = manifest.iloc[index].stage_name
fit = ps.load_fit(id)
session = cache.get_session(id)
cdf = psd.get_joint_table(fit,session)

fit = ps.plot_fit(id, cluster_labels=fit['all_clusters']['3'][1])
psd.mean_response_by_cluster(cdf,'3',session=id,stage = stage)
psd.running_behavior_by_cluster(cdf,'3',session=id,stage = stage)
psd.latency_behavior_by_cluster(cdf,3,session=id,stage = stage)
plt.figure(); plt.plot(fit['all_clusters']['3'][1])

# for many sessions
sessions, fits, cdfs = psd.build_multi_session_joint_table(ids[-3:],cache, manifest, use_all_clusters=True)
psd.mean_response_by_cluster(cdfs,'3',session=ids[-12:],stage = "")
psd.running_behavior_by_cluster(cdfs,'3',session=ids[-12:],stage = "")
psd.latency_behavior_by_cluster(cdfs,3,session=id,stage = stage)

# Is the next step here to do functional clustering?
# mean over all flash/cells?
# mean of mean-per-cell?
# Are we omitting omissions?
# how should we normalize across cells?
# image responsiveness (make roc type curves, fraction_significant vs fraction of cells)
# change responsiveness (make roc type curves)
# omission responsiveness (make roc type curves)
# response sparseness


