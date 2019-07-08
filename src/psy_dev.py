import psy_tools as ps
import matplotlib.pyplot as plt
from alex_utils import save
plt.ion()


ps.plot_session_summary(IDS)


# TODO
# 1. hierarchical clustering on weights across time, maybe do pca first
    # make score vs num-clusters plot
    # add clustering to plot_weights
        # pass in cluster labels as list of arrays, add row to include
# 2. fits over learning
# 3. make list of on-going issues to tackle later
# 4. make list of future extensions


# TODO ISSUES
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




# giant ID
# Giant stage names
# stages = ps.get_stage_names(IDS) # Takes Forever

IDS = [813083478, 888666715, 820307042, 820307518, 822656725, 815652334,
       822641265, 817267785, 873968820, 875045489, 833631914, 862848066,
       822647135, 878363070, 806456687, 891994418, 822028017, 817267860,
       822647116, 821011078, 825120601, 825130141, 823396897, 823392290,
       822024770, 806455766, 877697554, 840702910, 833629926, 892799212,
       891996193, 862023618, 841948542, 878358326, 834279512, 826585773,
       862848084, 826587940, 844395446, 842973730, 877022592, 877018118,
       843519218, 830093338, 826583436, 842975542, 847241639, 843520488,
       827236946, 848007790, 836258957, 847125577, 851060467, 831330404,
       850479305, 850489605, 845037476, 825623170, 863735602, 859147033,
       849199228, 846487947, 846490568, 849203586, 830697288, 849203565,
       830700781, 836912930, 830700800, 834279496, 836258936, 848692970,
       849204593, 848694045, 854703305, 896160394, 848694025, 851932055,
       848694639, 851056106, 836911939, 854703904, 866463736, 848697625,
       848697604, 853328133, 853328115, 869969393, 848698709, 805784331,
       837296345, 853962969, 853962951, 855577488, 855582961, 852691524,
       855582981, 868905381, 837729902, 856095742, 868911434, 856096766,
       869972431, 803736273, 775614751, 871159631, 838849908, 798403387,
       838849930, 877696762, 837297990, 864370674, 797255551, 806989729,
       799368904, 792813858, 796306435, 792816531, 799366517, 805100431,
       805784313, 796306417, 799368262, 865744231, 879332693, 794378505,
       792812544, 794381992, 796106321, 796105304, 778644591, 788490510,
       787498309, 783928214, 787501821, 788488596, 783927872, 784482326,
       788489531, 807752719, 807753318, 796106850, 807753334, 796105823,
       807753920, 873972085, 795075034, 795076128, 872432459, 795948257,
       795952488, 795952471, 790146945, 893831526, 790709081, 790149413,
       789359614, 791119849, 791453299, 791453282, 891054695, 880375092,
       893830418, 889771676, 891052180, 892805315, 889772922, 889777243,
       891067673, 889775742, 889775726, 886544609, 894726001, 894724592,
       894727297, 894724572, 895421128, 895422170, 810120743, 810119680,
       809497730, 808621015, 808619543, 808621034, 809501118, 808621958,
       811458048, 811456530, 880961028, 880374622, 882520593, 878363088,
       879331157, 884218326, 882935355, 881881193, 885061426, 884221469,
       885067016, 885067844, 885067826, 885933191]

for x in range(0,len(stages)):
    ps.plot_session_summary(stages[x],savefig=True,group_label=str(x))
    plt.close('all')


ps.plot_session_summary(stages[1]+stages[3],savefig=True,group_label="A_")
ps.plot_session_summary(stages[4],savefig=True,group_label="B1_")
ps.plot_session_summary(stages[6],savefig=True,group_label="B2_")
ps.plot_session_summary(stages[4]+stages[6],savefig=True,group_label="B_")


