import psy_tools as ps
import matplotlib.pyplot as plt
from alex_utils import save
plt.ion()

ps.plot_session_summary(IDS)

# Start of filtering my session type
stages = ps.get_stage_names(IDS) # Takes Forever
ps.plot_session_summary(stages[1]+stages[3],savefig=True,group_label="A_")
ps.plot_session_summary(stages[4],savefig=True,group_label="B1_")
ps.plot_session_summary(stages[6],savefig=True,group_label="B2_")
ps.plot_session_summary(stages[4]+stages[6],savefig=True,group_label="B_")

# TODO
# Next round of fits should: 
    # Fit every session in nick's database. Including the behavior only sessions which might need to be processed separately
    # put stage_name into metadata
    # save results as a dictionary rather than a list
    # finalize the dropout features
    # do clustering
# 0. fit for all ophys sessions for one mouse 
    # Filter the relevant sessions
    # Make a function that formats just one session (for behavior sessions)
    # Make a function that stiches sessions together 
    # set up fitting infrastructure
# 1. fits over learning
# 2. document psy_tools()
# 3. make list of on-going issues to tackle later
# 4. make list of future extensions

# TODO ISSUES
# include option for rolling dprime/hit/FA/miss/etc in clustering?
# Need better filtering of sessions, because there are some shitty ones still in there
# epochs of bad behavior dominate prior estimation. If you filter out very bad epochs, we'll get different priors
# have summary functions return mean+std, so you can compare across different groups
# add dprime trials to summary plots
# add dprime flashes to summary plots
# emperical/predicted accuracy
# format_session() is so slow
# need to deal with licking bouts that span two flashes
# more intelligent timing filters
# make fake data with different strategies: 
#   change bias/task/timing ratio
#   bias(-5:1:5) X task(-5:1:5) X timing(-5:1:5)
# examine effects of hyper-params
# Document that the aborted classification misses trials with dropped frames
# Document that bootstrapping isnt perfect because it doesnt sample the timing properly
# Sessions crash for unknown reason in compute_cross_validation_ypred, I cannot reproduce

import pandas as pd
behavior_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_sessions_to_load.h5', key='df')
all_flash_df = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_all_flash_df.h5', key='df')
behavior_psydata = ps.format_all_sessions(all_flash_df)
hyp2, evd2, wMode2, hess2, credibleInt2,weights2 = ps.fit_weights(behavior_psydata,TIMING4=True)
ypred2 = ps.compute_ypred(behavior_psydata, wMode2,weights2)
ps.plot_weights(session,wMode2, weights2,behavior_psydata,errorbar=credibleInt2, ypred = ypred2,validation=False,session_labels = behavior_sessions.stage_name.values)




from allensdk.brain_observatory.behavior import behavior_ophys_session as bos
from allensdk.brain_observatory.behavior import behavior_session as bs
from allensdk.brain_observatory.behavior import stimulus_processing
from allensdk.internal.api import behavior_ophys_api as boa
from allensdk.internal.api import behavior_lims_api as bla

# df with all the ophys and behavior sessions
vb_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/vb_sessions.h5', key='df')
mice = np.unique(vb_sessions['donor_id'])
vb_sessions[vb_sessions['donor_id'] == mice[7]]

def load_session(row):
    if pd.isnull(row['ophys_experiment_id']):
        api = bla.BehaviorLimsApi(int(row['behavior_session_id']))
        session_id = 'behavior_{}'.format(int(row['behavior_session_id']))
        session = bs.BehaviorSession(api)
    else:
        api = boa.BehaviorOphysLimsApi(int(row['ophys_experiment_id']))
        session_id = 'ophys_{}'.format(int(row['ophys_experiment_id']))
        session = bos.BehaviorOphysSession(api)
    return session, session_id











