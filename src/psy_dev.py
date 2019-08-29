import psy_tools as ps
import matplotlib.pyplot as plt
from alex_utils import *
plt.ion()

# LIST OF MICE AND SESSIONS TO FIT

session_ids = ps.get_session_ids() 
session_ids = ps.get_session_ids()
test_id = session_ids[-1]
test_session = ps.get_data(test_id)
test_psydata = ps.format_session(test_session)
test_psydata_old = ps.format_session_old(test_session)   




mice_ids = ps.get_mice_ids()
ps.process_mouse(mice_ids[0])
ps.process_session(session_ids[0])


### Dev below here
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











