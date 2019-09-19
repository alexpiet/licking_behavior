import psy_tools as ps
import psy_sdk_tools as psd
import matplotlib.pyplot as plt
from alex_utils import *
plt.ion()

# clustering + SDK
session_ids = ps.get_session_ids()

# For an individual session
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
cache = ps.get_cache()
manifest = ps.get_manifest()
sessions, fits, cdfs = psd.build_multi_session_joint_table(session_ids[0:40],cache, manifest, use_all_clusters=True,slim_df=True)
psd.mean_response_by_cluster(cdfs,'4',session=session_ids[0:40],stage = "")
psd.running_behavior_by_cluster(cdfs,'4',session=session_ids[0:40],stage = "")
psd.latency_behavior_by_cluster(cdfs,'4',session=session_ids[0:40],stage = stage)


# one off
for id in session_ids:
    print(id)
    try:
        psd.full_analysis(id)
    except:
        print(" crash")
    plt.close('all')



