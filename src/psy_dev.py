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
# 1. hierarchical clustering on weights across time, maybe do pca first
    # make score vs num-clusters plot
    # add clustering to plot_weights
        # pass in cluster labels as list of arrays, add row to include
# 2. fits over learning
# 3. make list of on-going issues to tackle later
# 4. make list of future extensions


# TODO ISSUES
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
from sklearn.cluster import k_means


numC=5
fig,ax = plt.subplots(nrows=numC,ncols=1)
scores = []
for j in range(0,numC):
    for i in range(0,4):
        ax[j].plot(ps.transform(wMode[i,:]))
        #ward = AgglomerativeClustering(n_clusters=2+j).fit(ps.transform(wMode.T))
    output = k_means(ps.transform(wMode.T),j+1)
    cp = np.where(~(np.diff(output[1]) == 0))[0]
    cp = np.concatenate([[0], cp, [len(output[1])]])
    colors = ['r','b','g','c','m','k','y']
    for i in range(0, len(cp)-1):
        ax[j].axvspan(cp[i],cp[i+1],color=colors[output[1][cp[i]+1]], alpha=0.1)
    ax[j].set_ylim(0,1)
    ax[j].set_xlim(0,len(wMode[0,:]))
    ax[j].set_ylabel(str(j+2)+" clusters")
    ax[j].set_xlabel('Flash #')
    scores.append(output[2])


plt.figure()
plt.plot(np.arange(1,j+2), scores/scores[0],'k-')







numC=8
plt.figure()
all_scores = []
for i in IDS:
    scores = []
    try:
        wMode = ps.get_all_weights([i])
    except:
        pass
    else:
        if not (type(wMode) == type(None)):
            for j in range(0,numC):
                output = k_means(ps.transform(wMode.T),j+1)
                scores.append(output[2])
            plt.plot(np.arange(1,j+2), scores/scores[0],'k-')
            all_scores.append(scores)


plt.figure()
for i in np.arange(0,len(all_scores)):
    plt.plot(np.arange(1,j+2), all_scores[i],'k-',alpha=0.3)


plt.figure()
for i in np.arange(0,len(all_scores)):
    plt.plot(np.arange(1,j+2), all_scores[i]/all_scores[i][0],'k-',alpha=0.3)


plt.ylabel('Normalized error')
plt.xlabel('number of clusters')



wMode = ps.get_all_weights(IDS)
numC = 10
full_scores=[]
for j in range(0,numC):
    output = k_means(ps.transform(wMode.T),j+1)
    full_scores.append(output[2])

plt.figure()
plt.plot(np.arange(0,numC)+1,full_scores,'ko')




