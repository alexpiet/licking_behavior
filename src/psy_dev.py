import psy_tools as ps
import matplotlib.pyplot as plt
from alex_utils import save
plt.ion()

IDS = [ 787498309, 796105823, 783927872,
        832117336, 842975542, 848006184,
        862023618, 867337243] 

experiment_id = IDS[5]
session = ps.get_data(experiment_id)
psydata = ps.format_session(session)
filename = '/home/alex.piet/codebase/behavior/psy_fits/' + str(experiment_id) 
hyp, evd, wMode, hess, credibleInt,weights = ps.fit_weights(psydata,TIMING4=True,OMISSIONS1=True)
ypred,ypred_each = ps.compute_ypred(psydata, wMode,weights)
ps.plot_weights(session,wMode, weights,psydata,errorbar=credibleInt, ypred = ypred,filename=filename)

# Takes forever
boots = ps.bootstrap(10, psydata, ypred, weights, wMode)
ps.plot_bootstrap(boots, hyp, weights, wMode, credibleInt,filename=filename)
models, labels = ps.dropout_analysis(psydata,TIMING5=True,OMISSIONS=True,OMISSIONS1=True)
ps.plot_dropout(models,labels,filename=filename)
save(filename+".pkl", [models, labels, boots, hyp, evd, wMode, hess, credibleInt, weights, ypred,psydata])
cross_results = ps.compute_cross_validation(psydata, hyp, weights,folds=10)
cv_pred = ps.compute_cross_validation_ypred(psydata, cross_results,ypred)


for id in IDS:
    try:
        print(id)
        ps.process_session(id)
        print('   complete '+str(id))
    except:
        print('   crash '+str(id))

ps.plot_session_summary_prior(IDS)
ps.plot_session_summary_dropout(IDS)
ps.plot_session_summary_weights(IDS)
ps.plot_session_summary_weight_range(IDS)
ps.plot_session_summary_weight_scatter(IDS)
ps.plot_session_summary_weight_avg_scatter(IDS)
ps.plot_session_summary_weight_trajectory(IDS)


# TODO
# 0. Dropout done with cross-validation
    # Dropout should include task1 vs task0 comparisons
# 1. Fit many sessions
    # fit more sessions
    # Save out metadata for session
    # figure out why sessions are crashing

# 2. Make summary figures
    # log-odds
    # epoch classification
    # avg trajectory for each weight (add average trace)
    # hierarchical clustering on weights across time (maybe do PCA first)
# 3. Make list of on-going issues to tackle later
# 4. Make list of future extensions

# add dprime trials
# add dprime flashes
# Cross validation
# emperical/predicted accuracy
# format_session() is so slow!
# need to deal with licking bouts that span two flashes
# more intelligent timing filters?
# make fake data with different strategies: 
#   change bias/task/timing ratio
#   bias(-5:1:5) X task(-5:1:5) X timing(-5:1:5)
# examine effects of hyper-params
# Document that the aborted classification misses trials with dropped frames
# Document that bootstrapping isnt perfect because it doesnt sample the timing properly

import pandas as pd
behavior_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_sessions_to_load.h5', key='df')
all_flash_df = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_all_flash_df.h5', key='df')
behavior_psydata = ps.format_all_sessions(all_flash_df)
hyp2, evd2, wMode2, hess2, credibleInt2,weights2 = ps.fit_weights(behavior_psydata,TIMING4=True)
ypred2 = ps.compute_ypred(behavior_psydata, wMode2,weights2)
ps.plot_weights(session,wMode2, weights2,behavior_psydata,errorbar=credibleInt2, ypred = ypred2,validation=False,session_labels = behavior_sessions.stage_name.values)



from sklearn.decomposition import PCA
pca = PCA()
pca.fit(wMode.T)
x = pca.fit_transform(wMode.T)
plt.figure()
plt.plot(x[:,0], x[:,1],'ko')
plt.plot(x[0,0], x[0,1],'ro')
plt.plot(x[-1,0], x[-1,1],'bo')
plt.plot(x[1500,0], x[1500,1],'go')
plt.plot(x[2700,0], x[2700,1],'go')
plt.plot(x[0:1500,0], x[0:1500,1],'ro')
plt.plot(x[1500:2700,0], x[1500:2700,1],'go')
plt.plot(x[2700:-1,0], x[2700:-1,1],'bo')
plt.xlim(-4,4)
plt.ylim(-4,4)





experiment_ids = [820298614, 813083478, 813070010, 825615139, 888666715, 820307042,
       820307518, 822656725, 815652334, 822641265, 817267785, 862848066,
       822015264, 822647135, 878363070, 823372519, 806456687, 822028017,
       817267860, 822647116, 821011078, 825120601, 825130141, 823396897,
       823392290, 822024770, 840702910, 862023618, 841948542, 826576503,
       826585773, 862848084, 827230913, 826583436, 827236946, 831330404,
       825623170, 846490568, 849203586, 830697288, 896160394, 848694025,
       866463736, 848697604, 855582981, 868911434, 856096766, 869972431,
       829408506, 817251835, 871159631, 877696762, 864370674, 865744231,
       879332693, 807752719, 816795311, 880375092, 889777243, 810120743,
       808619543, 811456530, 884218326, 882935355, 885061426, 884221469,
       885067826, 885933191]


