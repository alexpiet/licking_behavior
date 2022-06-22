import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import psy_general_tools as pgt



## Principal Component Analysis
#######################################################################


# TODO, Issue #159
def load_all_dropout(version=None):
    directory = pgt.get_directory(version,subdirectory='summary')
    dropout = load(directory+"all_dropouts.pkl")
    return dropout


# TODO, Issue #159
def get_all_dropout(IDS,version=None,verbose=False): 
    '''
        For each session in IDS, returns the vector of dropout scores for each model
    '''

    directory=pgt.get_directory(version,subdirectory='summary')

    all_dropouts = []
    hits = []
    false_alarms = []
    correct_reject = []
    misses = []
    bsids = []
    crashed = 0
    
    # Loop through IDS, add information from sessions above hit threshold
    for bsid in tqdm(IDS):
        try:
            fit = load_fit(bsid,version=version)
            dropout_dict = get_session_dropout(fit)
            dropout = [dropout_dict[x] for x in sorted(list(fit['weights'].keys()))] 
            all_dropouts.append(dropout)
            hits.append(np.sum(fit['psydata']['hits']))
            false_alarms.append(np.sum(fit['psydata']['false_alarms']))
            correct_reject.append(np.sum(fit['psydata']['correct_reject']))
            misses.append(np.sum(fit['psydata']['misses']))
            bsids.append(bsid)
        except:
            if verbose:
                print(str(bsid) +" crash")
            crashed +=1

    print(str(crashed) + " crashed")
    dropouts = np.stack(all_dropouts,axis=1)
    filepath = directory + "all_dropouts.pkl"
    save(filepath, dropouts)
    return dropouts,hits, false_alarms, misses,bsids, correct_reject

# TODO, Issue #190
def get_mice_weights(mice_ids,version=None,verbose=False,manifest = None):
    directory=pgt.get_directory(version)
    if manifest is None:
        manifest = pgt.get_ophys_manifest()
    mice_weights = []
    mice_good_ids = []
    crashed = 0
    # Loop through IDS
    for id in tqdm(mice_ids):
        this_mouse = []
        for sess in manifest.query('donor_id == @id').behavior_session_id.values:
            try:
                fit = load_fit(sess,version=version)
                this_mouse.append(np.mean(fit['wMode'],1))
            except:
                if verbose:
                    print("Mouse: "+str(id)+" session: "+str(sess) +" crash")
                crashed += 1
        if len(this_mouse) > 0:
            this_mouse = np.stack(this_mouse,axis=1)
            mice_weights.append(this_mouse)
            mice_good_ids.append(id)
    print()
    print(str(crashed) + " crashed")
    return mice_weights,mice_good_ids

# TODO, Issue #190
def get_mice_dropout(mice_ids,version=None,verbose=False,manifest=None):

    directory=pgt.get_directory(version)    
    if manifest is None:
        manifest = pgt.get_ophys_manifest()

    mice_dropouts = []
    mice_good_ids = []
    crashed = 0

    # Loop through IDS
    for id in tqdm(mice_ids):
        this_mouse = []
        for sess in manifest.query('donor_id ==@id')['behavior_session_id'].values:
            try:
                fit = load_fit(sess,version=version)
                dropout_dict = get_session_dropout(fit)
                dropout = [dropout_dict[x] for x in sorted(list(fit['weights'].keys()))] 
                this_mouse.append(dropout)
            except:
                if verbose:
                    print("Mouse: "+str(id)+" Session:"+str(sess)+" crash")
                crashed +=1
        if len(this_mouse) > 0:
            this_mouse = np.stack(this_mouse,axis=1)
            mice_dropouts.append(this_mouse)
            mice_good_ids.append(id)
    print()
    print(str(crashed) + " crashed")

    return mice_dropouts,mice_good_ids

# TODO, Issue #190
def PCA_dropout(ids,mice_ids,version,verbose=False,manifest=None,ms=2):
    dropouts, hits,false_alarms,misses,ids,correct_reject = get_all_dropout(ids,
        version,verbose=verbose)

    mice_dropouts, mice_good_ids = get_mice_dropout(mice_ids,
        version=version,verbose=verbose,
        manifest = manifest)

    fit = load_fit(ids[1],version=version)
    labels = sorted(list(fit['weights'].keys()))
    pca,dropout_dex,varexpl = PCA_on_dropout(dropouts, labels=labels,
        mice_dropouts=mice_dropouts,mice_ids=mice_good_ids, hits=hits,
        false_alarms=false_alarms, misses=misses,version=version, correct_reject = correct_reject,ms=ms)

    return dropout_dex,varexpl

# TODO, Issue #190
def PCA_on_dropout(dropouts,labels=None,mice_dropouts=None, mice_ids = None,
    hits=None,false_alarms=None, misses=None,version=None,fs1=12,fs2=12,
    filetype='.png',ms=2,correct_reject=None):

    directory=pgt.get_directory(version)
    if directory[-3:-1] == '12':
        sdex = 2
        edex = 6
    elif directory[-2] == '2':
        sdex = 2
        edex = 16
    elif directory[-2] == '4':
        sdex = 2
        edex = 18
    elif directory[-2] == '6':
        sdex = 2 
        edex = 6
    elif directory[-2] == '7':
        sdex = 2 
        edex = 6
    elif directory[-2] == '8':
        sdex = 2 
        edex = 6
    elif directory[-2] == '9':
        sdex = 2 
        edex = 6
    elif directory[-3:-1] == '10':
        sdex = 2
        edex = 6
    elif version == 20: 
        sdex = np.where(np.array(labels) == 'task0')[0][0]
        edex = np.where(np.array(labels) == 'timing1D')[0][0]
    dex = -(dropouts[sdex,:] - dropouts[edex,:])

    
    # Removing Bias from PCA
    dropouts = dropouts[1:,:]
    labels = labels[1:]

    # Do pca
    pca = PCA()
    pca.fit(dropouts.T)
    X = pca.transform(dropouts.T)
    
    fig,ax = plt.subplots(figsize=(6,4.5)) # FIG1
    fig=plt.gcf()
    ax = [plt.gca()]
    scat = ax[0].scatter(-X[:,0], X[:,1],c=dex,cmap='plasma')
    cbar = fig.colorbar(scat, ax = ax[0])
    cbar.ax.set_ylabel('Strategy Dropout Index',fontsize=fs2)
    ax[0].set_xlabel('Dropout PC 1',fontsize=fs1)
    ax[0].set_ylabel('Dropout PC 2',fontsize=fs1)
    ax[0].axis('equal')
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.tight_layout()   
    plt.savefig(directory+"figures_summary/dropout_pca"+filetype)
 
    plt.figure(figsize=(6,3))# FIG2
    fig=plt.gcf()
    ax.append(plt.gca())
    ax[1].axhline(0,color='k',alpha=0.2)
    for i in np.arange(0,len(dropouts)):
        if np.mod(i,2) == 0:
            ax[1].axvspan(i-.5,i+.5,color='k', alpha=0.1)
    pca1varexp = str(100*round(pca.explained_variance_ratio_[0],2))
    pca2varexp = str(100*round(pca.explained_variance_ratio_[1],2))
    ax[1].plot(-pca.components_[0,:],'ko-',label='PC1 '+pca1varexp+"%")
    ax[1].plot(-pca.components_[1,:],'ro-',label='PC2 '+pca2varexp+"%")
    ax[1].set_xlabel('Model Component',fontsize=12)
    ax[1].set_ylabel('% change in \n evidence',fontsize=12)
    ax[1].tick_params(axis='both',labelsize=10)
    ax[1].set_xticks(np.arange(0,len(dropouts)))
    if type(labels) is not type(None):    
        ax[1].set_xticklabels(labels,rotation=90)
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/dropout_pca_1.png")

    plt.figure(figsize=(5,4.5))# FIG3
    scat = plt.gca().scatter(-X[:,0],dex,c=dex,cmap='plasma')
    #cbar = plt.gcf().colorbar(scat, ax = plt.gca())
    #cbar.ax.set_ylabel('Task Dropout Index',fontsize=fs1)
    plt.gca().set_xlabel('Dropout PC 1',fontsize=fs1)
    plt.gca().set_ylabel('Strategy Dropout Index',fontsize=fs1)   
    plt.gca().axis('equal')
    plt.gca().tick_params(axis='both',labelsize=10)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/dropout_pca_3"+filetype)

    plt.figure(figsize=(5,4.5))# FIG4 
    ax = plt.gca()
    if type(mice_dropouts) is not type(None):
        ax.axhline(0,color='k',alpha=0.2)
        ax.set_xlabel('Individual Mice', fontsize=fs1)
        ax.set_ylabel('Strategy Dropout Index', fontsize=fs1)
        ax.set_xticks(range(0,len(mice_dropouts)))
        ax.set_ylim(-45,40)
        mean_drop = []
        for i in range(0, len(mice_dropouts)):
            mean_drop.append(-1*np.nanmean(mice_dropouts[i][sdex,:]-mice_dropouts[i][edex,:]))
        sortdex = np.argsort(np.array(mean_drop))
        mice_dropouts = [mice_dropouts[i] for i in sortdex]
        mean_drop = np.array(mean_drop)[sortdex]
        for i in range(0,len(mice_dropouts)):
            if np.mod(i,2) == 0:
                ax.axvspan(i-.5,i+.5,color='k', alpha=0.1)
            mouse_dex = -(mice_dropouts[i][sdex,:]-mice_dropouts[i][edex,:])
            ax.plot([i-0.5, i+0.5], [mean_drop[i],mean_drop[i]], 'k-',alpha=0.3)
            ax.scatter(i*np.ones(np.shape(mouse_dex)), mouse_dex,ms,c=mouse_dex,cmap='plasma',vmin=(dex).min(),vmax=(dex).max(),alpha=1)
        sorted_mice_ids = ["" for i in sortdex]
        ax.set_xticklabels(sorted_mice_ids,fontdict={'fontsize':10},rotation=90)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.xlim(-1,len(mice_dropouts))
    plt.savefig(directory+"figures_summary/dropout_pca_mice"+filetype)

    plt.figure(figsize=(5,4.5))
    ax = plt.gca()   
    ax.plot(pca.explained_variance_ratio_*100,'ko-')
    ax.set_xlabel('PC Dimension',fontsize=fs1)
    ax.set_ylabel('Explained Variance %',fontsize=fs1)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_var_expl"+filetype)

    fig, ax = plt.subplots(2,3,figsize=(10,6))
    #ax[0,0].axhline(0,color='k',alpha=0.2)
    #ax[0,0].axvline(0,color='k',alpha=0.2)
    ax[0,0].scatter(-X[:,0], dex,c=dex,cmap='plasma')
    ax[0,0].set_xlabel('Dropout PC 1',fontsize=fs2)
    ax[0,0].set_ylabel('Strategy Dropout Index',fontsize=fs2)
    ax[0,1].plot(pca.explained_variance_ratio_*100,'ko-')
    ax[0,1].set_xlabel('PC Dimension',fontsize=fs2)
    ax[0,1].set_ylabel('Explained Variance %',fontsize=fs2)

    if type(mice_dropouts) is not type(None):
        ax[1,0].axhline(0,color='k',alpha=0.2)
        ax[1,0].set_ylabel('Strategy Dropout Index', fontsize=12)
        ax[1,0].set_xticks(range(0,len(mice_dropouts)))
        ax[1,0].set_ylim(-45,40)
        mean_drop = []
        for i in range(0, len(mice_dropouts)):
            mean_drop.append(-1*np.nanmean(mice_dropouts[i][sdex,:]-mice_dropouts[i][edex,:]))
        sortdex = np.argsort(np.array(mean_drop))
        mice_dropouts = [mice_dropouts[i] for i in sortdex]
        mean_drop = np.array(mean_drop)[sortdex]
        for i in range(0,len(mice_dropouts)):
            if np.mod(i,2) == 0:
                ax[1,0].axvspan(i-.5,i+.5,color='k', alpha=0.1)
            mouse_dex = -(mice_dropouts[i][sdex,:]-mice_dropouts[i][edex,:])
            ax[1,0].plot([i-0.5, i+0.5], [mean_drop[i],mean_drop[i]], 'k-',alpha=0.3)
            ax[1,0].scatter(i*np.ones(np.shape(mouse_dex)), mouse_dex,c=mouse_dex,cmap='plasma',vmin=(dex).min(),vmax=(dex).max(),alpha=1)
        sorted_mice_ids = [mice_ids[i] for i in sortdex]
        ax[1,0].set_xticklabels(sorted_mice_ids,fontdict={'fontsize':10},rotation=90)
    if type(hits) is not type(None):
        ax[1,1].scatter(dex, hits,c=dex,cmap='plasma')
        ax[1,1].set_ylabel('Hits/session',fontsize=12)
        ax[1,1].set_xlabel('Strategy Dropout Index',fontsize=12)
        ax[1,1].axvline(0,color='k',alpha=0.2)
        ax[1,1].set_xlim(-45,40)
        ax[1,1].set_ylim(bottom=0)

        ax[0,2].scatter(dex, false_alarms,c=dex,cmap='plasma')
        ax[0,2].set_ylabel('FA/session',fontsize=12)
        ax[0,2].set_xlabel('Strategy Dropout Index',fontsize=12)
        ax[0,2].axvline(0,color='k',alpha=0.2)
        ax[0,2].set_xlim(-45,40)
        ax[0,2].set_ylim(bottom=0)


        ax[1,2].scatter(dex, misses,c=dex,cmap='plasma')
        ax[1,2].set_ylabel('Miss/session',fontsize=12)
        ax[1,2].set_xlabel('Strategy Dropout Index',fontsize=12)
        ax[1,2].axvline(0,color='k',alpha=0.2)
        ax[1,2].set_xlim(-45,40)
        ax[1,2].set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/dropout_pca_2.png")

    plt.figure(figsize=(5,4.5))
    ax = plt.gca() 
    ax.scatter(dex, hits,c=dex,cmap='plasma')
    ax.set_ylabel('Hits/session',fontsize=fs1)
    ax.set_xlabel('Strategy Dropout Index',fontsize=fs1)
    ax.axvline(0,color='k',alpha=0.2)
    ax.set_xlim(-45,40)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_hits"+filetype)


    plt.figure(figsize=(5,4.5))
    ax = plt.gca()
    ax.scatter(dex, false_alarms,c=dex,cmap='plasma')
    ax.set_ylabel('FA/session',fontsize=fs1)
    ax.set_xlabel('Strategy Dropout Index',fontsize=fs1)
    ax.axvline(0,color='k',alpha=0.2)
    ax.set_xlim(-45,40)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_fa"+filetype)



    plt.figure(figsize=(5,4.5))
    ax = plt.gca() 
    ax.scatter(dex, misses,c=dex,cmap='plasma')
    ax.set_ylabel('Miss/session',fontsize=fs1)
    ax.set_xlabel('Strategy Dropout Index',fontsize=fs1)
    ax.axvline(0,color='k',alpha=0.2)
    ax.set_xlim(-45,40)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_miss"+filetype)

    plt.figure(figsize=(5,4.5))
    ax = plt.gca() 
    ax.scatter(dex, correct_reject,c=dex,cmap='plasma')
    ax.set_ylabel('CR/session',fontsize=fs1)
    ax.set_xlabel('Strategy Dropout Index',fontsize=fs1)
    ax.axvline(0,color='k',alpha=0.2)
    ax.set_xlim(-45,40)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.savefig(directory+"figures_summary/dropout_pca_cr"+filetype)

    varexpl = 100*round(pca.explained_variance_ratio_[0],2)
    return pca,dex,varexpl

# TODO, Issue #190
def PCA_weights(ids,mice_ids,version=None,verbose=False,manifest = None):
    directory=pgt.get_directory(version)
    #all_weights,good_ids =plot_session_summary_weights(ids,return_weights=True,version=version)
    plot_session_summary_weights(ids,return_weights=True,version=version)
    x = np.vstack(all_weights)

    fit = load_fit(ids[np.where(good_ids)[0][0]],version=version)
    weight_names = sorted(list(fit['weights'].keys()))
    task_index = np.where(np.array(weight_names) == 'task0')[0][0]
    timing_index = np.where(np.array(weight_names) == 'timing1D')[0][0]
    task = x[:,np.where(np.array(weight_names) == 'task0')[0][0]]
    timing = x[:,np.where(np.array(weight_names) == 'timing1D')[0][0]]

    dex = task-timing
    pca = PCA()
    pca.fit(x)
    X = pca.transform(x)
    plt.figure(figsize=(4,2.9))
    scat = plt.gca().scatter(X[:,0],X[:,1],c=dex,cmap='plasma')
    cbar = plt.gcf().colorbar(scat, ax = plt.gca())
    cbar.ax.set_ylabel('Strategy Weight Index',fontsize=12)
    plt.gca().set_xlabel('Weight PC 1 - '+str(100*round(pca.explained_variance_ratio_[0],2))+"%",fontsize=12)
    plt.gca().set_ylabel('Weight PC 2 - '+str(100*round(pca.explained_variance_ratio_[1],2))+"%",fontsize=12)
    plt.gca().axis('equal')   
    plt.gca().tick_params(axis='both',labelsize=10)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/weight_pca_1.png")

    plt.figure(figsize=(4,2.9))
    scat = plt.gca().scatter(X[:,0],dex,c=dex,cmap='plasma')
    cbar = plt.gcf().colorbar(scat, ax = plt.gca())
    cbar.ax.set_ylabel('Strategy Weight Index',fontsize=12)
    plt.gca().set_xlabel('Weight PC 1',fontsize=12)
    plt.gca().set_ylabel('Strategy Weight Index',fontsize=12)
    plt.gca().axis('equal')
    plt.gca().tick_params(axis='both',labelsize=10)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/weight_pca_2.png")   

    plt.figure(figsize=(6,3))
    fig=plt.gcf()
    ax =plt.gca()
    ax.axhline(0,color='k',alpha=0.2)
    for i in np.arange(0,np.shape(x)[1]):
        if np.mod(i,2) == 0:
            ax.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    pca1varexp = str(100*round(pca.explained_variance_ratio_[0],2))
    pca2varexp = str(100*round(pca.explained_variance_ratio_[1],2))
    ax.plot(pca.components_[0,:],'ko-',label='PC1 '+pca1varexp+"%")
    ax.plot(pca.components_[1,:],'ro-',label='PC2 '+pca2varexp+"%")
    ax.set_xlabel('Model Component',fontsize=12)
    ax.set_ylabel('Avg Weight',fontsize=12)
    ax.tick_params(axis='both',labelsize=10)
    ax.set_xticks(np.arange(0,np.shape(x)[1]))
    weights_list = get_weights_list(fit['weights'])
    labels = pgt.get_clean_string(weights_list)    
    ax.set_xticklabels(labels,rotation=90)
    ax.legend()
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/weight_pca_3.png")

    _, hits,false_alarms,misses,ids = get_all_dropout(ids,version=version,verbose=verbose)
    mice_weights, mice_good_ids = get_mice_weights(mice_ids, version=version,verbose=verbose,manifest = manifest)

    fig, ax = plt.subplots(2,3,figsize=(10,6))
    ax[0,0].scatter(X[:,0], dex,c=dex,cmap='plasma')
    ax[0,0].set_xlabel('Weight PC 1',fontsize=12)
    ax[0,0].set_ylabel('Strategy Weight Index',fontsize=12)
    ax[0,1].plot(pca.explained_variance_ratio_*100,'ko-')
    ax[0,1].set_xlabel('PC Dimension',fontsize=12)
    ax[0,1].set_ylabel('Explained Variance %',fontsize=12)

    ax[1,0].axhline(0,color='k',alpha=0.2)
    ax[1,0].set_ylabel('Strategy Weight Index', fontsize=12)
    ax[1,0].set_xticks(range(0,len(mice_good_ids)))
    ax[1,0].set_ylim(-8,8)
    mean_weight = []
    for i in range(0, len(mice_good_ids)):
        this_weight = np.mean(mice_weights[i],1)
        mean_weight.append(this_weight[task_index] -this_weight[timing_index])
    sortdex = np.argsort(np.array(mean_weight))
    mice_weights_sorted = [mice_weights[i] for i in sortdex]
    mean_weight = np.array(mean_weight)[sortdex]
    for i in range(0,len(mice_good_ids)):
        if np.mod(i,2) == 0:
            ax[1,0].axvspan(i-.5,i+.5,color='k', alpha=0.1)
        this_mouse_weights = mice_weights_sorted[i][task_index,:] - mice_weights_sorted[i][timing_index,:]
        ax[1,0].plot([i-0.5,i+0.5],[mean_weight[i],mean_weight[i]],'k-',alpha=0.3)
        ax[1,0].scatter(i*np.ones(np.shape(this_mouse_weights)), this_mouse_weights,c=this_mouse_weights,cmap='plasma',vmin=(dex).min(),vmax=(dex).max(),alpha=1)
    sorted_mice_ids = [mice_good_ids[i] for i in sortdex]
    ax[1,0].set_xticklabels(sorted_mice_ids,fontdict={'fontsize':10},rotation=90) 
    ax[1,1].scatter(dex, hits,c=dex,cmap='plasma')
    ax[1,1].set_ylabel('Hits/session',fontsize=12)
    ax[1,1].set_xlabel('Strategy Weight Index',fontsize=12)
    ax[1,1].axvline(0,color='k',alpha=0.2)
    ax[1,1].set_xlim(-8,8)
    ax[1,1].set_ylim(bottom=0)

    ax[0,2].scatter(dex, false_alarms,c=dex,cmap='plasma')
    ax[0,2].set_ylabel('FA/session',fontsize=12)
    ax[0,2].set_xlabel('Strategy Weight Index',fontsize=12)
    ax[0,2].axvline(0,color='k',alpha=0.2)
    ax[0,2].set_xlim(-8,8)
    ax[0,2].set_ylim(bottom=0)

    ax[1,2].scatter(dex, misses,c=dex,cmap='plasma')
    ax[1,2].set_ylabel('Miss/session',fontsize=12)
    ax[1,2].set_xlabel('Strategy Weight Index',fontsize=12)
    ax[1,2].axvline(0,color='k',alpha=0.2)
    ax[1,2].set_xlim(-8,8)
    ax[1,2].set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(directory+"figures_summary/weight_pca_4.png")

    varexpl =100*round(pca.explained_variance_ratio_[0],2)
    return dex, varexpl



# TODO, Issue #190
def PCA_analysis(ids, mice_ids,version,manifest=None,savefig=False, group=None):
    # PCA on dropouts
    drop_dex,drop_varexpl = PCA_dropout(ids,mice_ids,version,manifest=manifest)

    # PCA on weights
    weight_dex,weight_varexpl = PCA_weights(ids,mice_ids,version,manifest=manifest)
   
    # Compare PCA on weights and dropouts
    fig, ax = plt.subplots(figsize=(5,4.5))
    scat = ax.scatter(weight_dex,drop_dex,c=weight_dex, cmap='plasma')
    ax.set_xlabel('Task Weight Index' ,fontsize=24)
    ax.set_ylabel('Task Dropout Index',fontsize=24)
    #cbar = plt.gcf().colorbar(scat, ax = ax)
    #cbar.ax.set_ylabel('Task Weight Index',fontsize=20)
    ax.tick_params(axis='both',labelsize=10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()

    if savefig:
        directory=pgt.get_directory(version) 
        plt.savefig(directory+"figures_summary/dropout_vs_weight_pca_1.svg")





## Event Triggered Analysis
#######################################################################
def triggered_analysis(ophys, version=None,triggers=['hit','miss'],dur=50,responses=['lick_bout_rate']):
    # Iterate over sessions

    plt.figure()
    for trigger in triggers:
        for response in responses:
            stas =[]
            skipped = 0
            for index, row in ophys.iterrows():
                try:
                    stas.append(session_triggered_analysis(row, trigger, response,dur))
                except:
                    pass
            mean = np.nanmean(stas,0)
            n=np.shape(stas)[0]
            std = np.nanstd(stas,0)/np.sqrt(n)

            plt.plot(mean,label=response+' by '+trigger)
            plt.plot(mean+std,'k')
            plt.plot(mean-std,'k')       
    plt.legend()

def session_triggered_analysis(ophys_row,trigger,response, dur):
    indexes = np.where(ophys_row[trigger] ==1)[0]
    vals = []
    for index in indexes:
        vals.append(get_aligned(ophys_row[response],index, length=dur))
    if len(vals) >1:
        mean= np.mean(np.vstack(vals),0)
        mean = mean - mean[0]
    else:
        mean = np.array([np.nan]*dur)
    return mean

def plot_triggered_analysis(row,trigger,responses,dur):
    plt.figure()
    for response in responses:
        sta = session_triggered_analysis(row,trigger, response,dur)
        plt.plot(sta, label=response)
        #plt.plot(sta+sem1,'k')
        #plt.plot(sta-sem1,'k')       
   
    plt.axhline(0,color='k',linestyle='--',alpha=.5) 
    plt.ylabel('change relative to hit/FA')
    plt.xlabel(' image #') 
    plt.legend()

def get_aligned(vector, start, length=4800):

    if len(vector) >= start+length:
        aligned= vector[start:start+length]
    else:
        aligned = np.concatenate([vector[start:], [np.nan]*(start+length-len(vector))])
    return aligned





