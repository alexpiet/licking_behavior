import psy_tools as ps
import matplotlib.pyplot as plt
plt.ion()

IDS = [ 787498309, 796105823, 783927872,
        832117336, 842975542, 848006184,
        862023618, 867337243] 

experiment_id = IDS[5]
session = ps.get_data(experiment_id)
psydata = ps.format_session(session)
hyp, evd, wMode, hess, credibleInt,weights = ps.fit_weights(psydata,TIMING4=True)
ypred,ypred_each = ps.compute_ypred(psydata, wMode,weights)
ps.plot_weights(session,wMode, weights,psydata,errorbar=credibleInt, ypred = ypred)

# Takes forever
boots = ps.bootstrap(10, psydata, ypred, weights, wMode)
ps.plot_bootstrap(boots, hyp, weights, wMode, credibleInt)
models, labels = ps.dropout_analysis(psydata,TIMING5=True,OMISSIONS=True)
ps.plot_dropout(models,labels)

## TODO
# Document that the aborted classification misses trials with dropped frames
# Document that bootstrapping isnt perfect because it doesnt sample the timing properly

# add dprime trials
# add dprime flashes
# debug omissions filter, should we regress on omitted flash, or omitted +1?
# omissions on learning
# run many sessions, and create summary statistics

# format_session() is so slow!
# need to deal with licking bouts that span two flashes
# more intelligent timing filters?
# make fake data with different strategies: 
#   change bias/task ratio
#   bias(-5:1:5) X task(-5:1:5)
# examine effects of hyper-params

import pandas as pd
behavior_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_sessions_to_load.h5', key='df')
all_flash_df = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_all_flash_df.h5', key='df')
behavior_psydata = ps.format_all_sessions(all_flash_df)
hyp2, evd2, wMode2, hess2, credibleInt2,weights2 = ps.fit_weights(behavior_psydata)
ypred2 = ps.compute_ypred(behavior_psydata, wMode2,weights2)
ps.plot_weights(session,wMode2, weights2,behavior_psydata,errorbar=credibleInt2, ypred = ypred2,validation=False,session_labels = behavior_sessions.stage_name.values)


