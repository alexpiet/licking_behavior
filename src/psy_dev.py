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

ps.plot_session_summary(IDS)

# TODO
# 0. Update dropout summary plot to include cross validation score (make second version)
# 1. Hierarchical Clustering on weights across time, maybe do PCA first
    # add clustering to plot_weights
        # pass in cluster labels as list of arrays, add row to include
# 2. Fits over learning
# 3. Make list of on-going issues to tackle later
# 4. Make list of future extensions


# TODO ISSUES
# add dprime trials to summary plots
# add dprime flashes to summary plots
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
# Sessions crash for unknown reason in compute_cross_validation_ypred, I cannot reproduce

import pandas as pd
behavior_sessions = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_sessions_to_load.h5', key='df')
all_flash_df = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190626_all_flash_df.h5', key='df')
behavior_psydata = ps.format_all_sessions(all_flash_df)
hyp2, evd2, wMode2, hess2, credibleInt2,weights2 = ps.fit_weights(behavior_psydata,TIMING4=True)
ypred2 = ps.compute_ypred(behavior_psydata, wMode2,weights2)
ps.plot_weights(session,wMode2, weights2,behavior_psydata,errorbar=credibleInt2, ypred = ypred2,validation=False,session_labels = behavior_sessions.stage_name.values)

#### Fucking with PCA
weights = ps.get_all_weights(IDS, directory='/home/alex.piet/codebase/behavior/psy_fits/first/')
from sklearn.decomposition import PCA

# Do PCA on weights in linear space
pca = PCA()
pca.fit(weights.T)
x = pca.fit_transform(weights.T)

fig, ax = plt.subplots(nrows=2,ncols=2)
for i in range(0,2):
    for j in range(0,2):
        ax[i,j].hist(x[:,i+j+j],bins=1000)

fig, ax = plt.subplots(nrows=2,ncols=2)
for i in range(0,2):
    for j in range(0,2):
        ax[i,j].hist(weights[i+j+j,:],bins=1000)



# Do PCA on weights in nonlinear space
pca = PCA()
pca.fit(ps.transform(weights.T))
x = pca.fit_transform(ps.transform(weights.T))

fig, ax = plt.subplots(nrows=2,ncols=2)
for i in range(0,2):
    for j in range(0,2):
        ax[i,j].hist(x[:,i+j+j],bins=1000)

fig, ax = plt.subplots(nrows=2,ncols=2)
for i in range(0,2):
    for j in range(0,2):
        ax[i,j].hist(ps.transform(weights[i+j+j,:]),bins=1000)


fig = plt.figure()
plt.hexbin(x[:,0],x[:,1],gridsize=30,cmap='Blues')
cb = plt.colorbar(label='count')


##### Fucking with Clustering

weights = ps.get_all_weights([IDS[3]], directory='/home/alex.piet/codebase/behavior/psy_fits/first/')
from sklearn.cluster import AgglomerativeClustering


numC = 6
fig,ax = plt.subplots(nrows=numC,ncols=1)
for j in range(0,numC):
    for i in range(0,4):
        ax[j].plot(ps.transform(wMode[i,:]))
    ward = AgglomerativeClustering(n_clusters=2+j).fit(wMode.T)
    cp = np.where(~(np.diff(ward.labels_) == 0))[0]
    cp = np.concatenate([[0], cp, [len(ward.labels_)]])
    colors = ['r','b','g','c','m','k','y']
    for i in range(0, len(cp)-1):
        ax[j].axvspan(cp[i],cp[i+1],color=colors[ward.labels_[cp[i]+1]], alpha=0.1)
    ax[j].set_ylim(0,1)
    ax[j].set_xlim(0,len(wMode[0,:]))
    ax[j].set_ylabel(str(j+2)+" clusters")
    ax[j].set_xlabel('Flash #')

fig,ax = plt.subplots(nrows=numC,ncols=1)
for j in range(0,numC):
    for i in range(0,4):
        ax[j].plot(ps.transform(wMode[i,:]))
    ward = AgglomerativeClustering(n_clusters=2+j).fit(ps.transform(wMode.T))
    cp = np.where(~(np.diff(ward.labels_) == 0))[0]
    cp = np.concatenate([[0], cp, [len(ward.labels_)]])
    colors = ['r','b','g','c','m','k','y']
    for i in range(0, len(cp)-1):
        ax[j].axvspan(cp[i],cp[i+1],color=colors[ward.labels_[cp[i]+1]], alpha=0.1)
    ax[j].set_ylim(0,1)
    ax[j].set_xlim(0,len(wMode[0,:]))
    ax[j].set_ylabel(str(j+2)+" clusters")
    ax[j].set_xlabel('Flash #')



