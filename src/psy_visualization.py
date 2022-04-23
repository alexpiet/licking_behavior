import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import psy_style as pstyle
import psy_timing_tools as pt
import psy_metrics_tools as pm
import psy_tools as ps
import psy_general_tools as pgt


def plot_session_summary(summary_table,version=None,savefig=False,group_label="",nel=4):
    '''
        Makes a series of summary plots for all the IDS
    '''
    ids = summary_table['behavior_session_id'].values
    directory=pgt.get_directory(version)
    plot_session_summary_priors(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout(IDS,version=version,cross_validation=False,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout(IDS,version=version,cross_validation=True,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout_scatter(IDS, version=version, savefig=savefig, group_label=group_label); plt.close('all')
    plot_session_summary_weights(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_weight_range(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_weight_scatter(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter_task0(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter_hits(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter_miss(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter_false_alarms(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_trajectory(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_logodds(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_correlation(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_roc(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_static_comparison(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')

 
def get_session_summary(behavior_session_id,cross_validation_dropout=True,model_evidence=False,version=None,hit_threshold=0):
    '''
        Extracts useful summary information about each fit
        if cross_validation_dropout, then uses the dropout analysis where each reduced model is cross-validated
    '''
    directory = pgt.get_directory(version)
    fit = ps.load_fit(behavior_session_id, version=version)

    if type(fit) is not dict:
        labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
        fit = dict((x,y) for x,y in zip(labels, fit))

    if np.sum(fit['psydata']['hits']) < hit_threshold:
        raise Exception('Below hit threshold')    

    # compute statistics
    dropout = ps.get_session_dropout(fit,cross_validation=cross_validation_dropout)
    avgW = np.mean(fit['wMode'],1)
    rangeW = np.ptp(fit['wMode'],1)
    labels =sorted(list(fit['models'].keys()))
    return fit['hyp']['sigma'],fit['weights'],dropout,labels, avgW, rangeW,fit['wMode'],fit



def plot_session_summary_priors(IDS,version=None,savefig=False,group_label="",fs1=12,fs2=12,filetype='.png'):
    '''
        Make a summary plot of the priors on each feature
    '''
    directory=pgt.get_directory(version)

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    alld = []
    counter = 0
    for id in tqdm(IDS):
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass 
        else:
            sigmas = session_summary[0]
            weights = session_summary[1]
            ax.plot(np.arange(0,len(sigmas)),sigmas, 'o',alpha = 0.5)
            plt.yscale('log')
            plt.ylim(0.0001, 20)
            ax.set_xticks(np.arange(0,len(sigmas)))
            weights_list = ps.clean_weights(ps.get_weights_list(weights))
            ax.set_xticklabels(weights_list,fontsize=fs2,rotation=90)
            plt.ylabel('Smoothing Prior, $\sigma$\n <-- smooth           variable --> ',fontsize=fs1)
            counter +=1
            alld.append(sigmas)            

    if counter == 0:
        print('NO DATA')
        return
    alld = np.mean(np.vstack(alld),0)
    for i in np.arange(0, len(sigmas)):
        ax.plot([i-.25, i+.25],[alld[i],alld[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    ax.axhline(0.001,color='k',alpha=0.2)
    ax.axhline(0.01,color='k',alpha=0.2)
    ax.axhline(0.1,color='k',alpha=0.2)
    ax.axhline(1,color='k',alpha=0.2)
    ax.axhline(10,color='k',alpha=0.2)
    plt.yticks(fontsize=fs2-4,rotation=90)
    ax.xaxis.tick_top()
    ax.set_xlim(xmin=-.5)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"prior"+filetype)



def compute_model_prediction_correlation(fit,fit_mov=50,data_mov=50,plot_this=False,cross_validation=True):
    '''
        Computes the R^2 value between the model predicted licking probability, and the smoothed data lick rate.
        The data is smoothed over data_mov flashes. The model is smoothed over fit_mov flashes. Both smoothings uses a moving _mean within that range. 
        if plot_this, then the two smoothed traces are plotted
        if cross_validation, then uses the cross validated model prediction, and not the training set predictions
        Returns, the r^2 value.
    '''
    if cross_validation:
        data = copy.copy(fit['psydata']['y']-1)
        model = copy.copy(fit['cv_pred'])
    else:
        data = copy.copy(fit['psydata']['y']-1)
        model = copy.copy(fit['ypred'])
    data_smooth = pgt.moving_mean(data,data_mov)
    ypred_smooth = pgt.moving_mean(model,fit_mov)

    minlen = np.min([len(data_smooth), len(ypred_smooth)])
    if plot_this:
        plt.figure()
        plt.plot(ypred_smooth, 'k')
        plt.plot(data_smooth,'b')
    return round(np.corrcoef(ypred_smooth[0:minlen], data_smooth[0:minlen])[0,1]**2,2)




def plot_session_summary_correlation(IDS,version=None,savefig=False,group_label="",verbose=True):
    '''
        Make a summary plot of the priors on each feature
    '''
    directory=pgt.get_directory(version)
    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    scores = []
    ids = []
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            fit = session_summary[7]
            r2 = compute_model_prediction_correlation(fit,fit_mov=25,data_mov=25,plot_this=False,cross_validation=True)
            scores.append(r2)
            ids.append(id)
            counter +=1

    if counter == 0:
        print('NO DATA')
        return

    ax.hist(np.array(scores),bins=50)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('$R^2$', fontsize=12)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    meanscore = np.median(np.array(scores))
    ax.plot(meanscore, ax.get_ylim()[1],'rv')
    ax.axvline(meanscore,color='r', alpha=0.3)
    ax.set_xlim(0,1)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"correlation.png")
    if verbose:
        median = np.argsort(np.array(scores))[len(scores)//2]
        best = np.argmax(np.array(scores))
        worst = np.argmin(np.array(scores)) 
        print('R^2 Correlation:')
        print('Worst  Session: ' + str(ids[worst]) + " " + str(scores[worst]))
        print('Median Session: ' + str(ids[median]) + " " + str(scores[median]))
        print('Best   Session: ' + str(ids[best]) + " " + str(scores[best]))      
    return scores, ids 


def plot_session_summary_dropout(IDS,version=None,cross_validation=True,savefig=False,group_label="",model_evidence=False,fs1=12,fs2=12,filetype='.png'):
    '''
        Make a summary plot showing the fractional change in either model evidence (not cross-validated), or log-likelihood (cross-validated)
    '''
    directory=pgt.get_directory(version)
    
    # make figure    
    fig,ax = plt.subplots(figsize=(7.2,6))
    alld = []
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    if cross_validation:
        plt.ylabel('% Change in CV Likelihood \n <-- Worse Fit',fontsize=fs1)
    else:
        plt.ylabel('% Change in Likelihood \n <-- Worse Fit',fontsize=fs1)

    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version, cross_validation_dropout=cross_validation)
        except:
            pass
        else:
            dropout_dict = session_summary[2]
            labels  = session_summary[3]
            dropout = [dropout_dict[x] for x in labels[1:]]
            ax.plot(np.arange(0,len(dropout)),dropout, 'o',alpha=0.5)
            alld.append(dropout)
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    alld = np.mean(np.vstack(alld),0)
    plt.yticks(fontsize=fs2-4,rotation=90)
    for i in np.arange(0, len(dropout)):
        ax.plot([i-.25, i+.25],[alld[i],alld[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    ax.set_xticks(np.arange(0,len(dropout)))
    ax.set_xticklabels(ps.clean_weights(labels[1:]),fontsize=fs2, rotation = 90)
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.xlim(-0.5,len(dropout) - 0.5)
    plt.ylim(-80,5)
    if savefig:
        if cross_validation:
            plt.savefig(directory+"figures_summary/summary_"+group_label+"dropout_cv"+filetype)
        else:
            plt.savefig(directory+"figures_summary/summary_"+group_label+"dropout"+filetype)


def plot_session_summary_weights(IDS,version=None, savefig=False,group_label="",return_weights=False,fs1=12,fs2=12,filetype='.svg',hit_threshold=0):
    '''
        Makes a summary plot showing the average weight value for each session
    '''
    directory=pgt.get_directory(version)

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    all_weights = []
    good = []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version,hit_threshold=hit_threshold)
        except:
            good.append(False)
        else:
            good.append(True)
            avgW = session_summary[4]
            weights  = session_summary[1]
            ax.plot(np.arange(0,len(avgW)),avgW, 'o',alpha=0.5)
            ax.set_xticks(np.arange(0,len(avgW)))
            plt.ylabel('Avg. Weights across each session',fontsize=fs1)

            all_weights.append(avgW)
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    allW = np.mean(np.vstack(all_weights),0)
    for i in np.arange(0, len(avgW)):
        ax.plot([i-.25, i+.25],[allW[i],allW[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    weights_list = ps.get_weights_list(weights)
    ax.set_xticklabels(ps.clean_weights(weights_list),fontsize=fs2, rotation = 90)
    ax.xaxis.tick_top()
    plt.yticks(fontsize=fs2-4,rotation=90)
    plt.tight_layout()
    plt.xlim(-0.5,len(avgW) - 0.5)
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weights"+filetype)
    if return_weights:
        return all_weights, good


def plot_session_summary_weight_range(IDS,version=None,savefig=False,group_label=""):
    '''
        Makes a summary plot showing the range of each weight across each session
    '''
    directory=pgt.get_directory(version)

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    allW = None
    counter = 0
    ax.axhline(0,color='k',alpha=0.2)
    all_range = []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            rangeW = session_summary[5]
            weights  = session_summary[1]
            ax.plot(np.arange(0,len(rangeW)),rangeW, 'o',alpha=0.5)
            ax.set_xticks(np.arange(0,len(rangeW)))
            plt.ylabel('Range of Weights across each session',fontsize=12)
            all_range.append(rangeW)    
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    allW = np.mean(np.vstack(all_range),0)
    for i in np.arange(0, len(rangeW)):
        ax.plot([i-.25, i+.25],[allW[i],allW[i]], 'k-',lw=3)
        if np.mod(i,2) == 0:
            plt.axvspan(i-.5,i+.5,color='k', alpha=0.1)
    weights_list = ps.get_weights_list(weights)
    ax.set_xticklabels(ps.clean_weights(weights_list),fontsize=12, rotation = 90)
    ax.xaxis.tick_top()
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.xlim(-0.5,len(rangeW) - 0.5)
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_range.png")


def plot_session_summary_weight_scatter(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against each other weight, plotting the average weight for each session
    '''
    directory = pgt.get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=nel,ncols=nel,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = ps.get_weights_list(weights)
            for i in np.arange(0,np.shape(W)[0]):
                if i < np.shape(W)[0]-1:
                    for j in np.arange(1, i+1):
                        ax[i,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                        ax[i,j-1].set_xticks([])
                        ax[i,j-1].set_yticks([])
                        for spine in ax[i,j-1].spines.values():
                            spine.set_visible(False)
                for j in np.arange(i+1,np.shape(W)[0]):
                    ax[i,j-1].axvline(0,color='k',alpha=0.05)
                    ax[i,j-1].axhline(0,color='k',alpha=0.05)
                    ax[i,j-1].plot(W[j,:], W[i,:],'o', alpha=0.01)
                    ax[i,j-1].set_xlabel(weights_list[j],fontsize=12)
                    ax[i,j-1].set_ylabel(weights_list[i],fontsize=12)
                    ax[i,j-1].xaxis.set_tick_params(labelsize=12)
                    ax[i,j-1].yaxis.set_tick_params(labelsize=12)
            counter +=1
    plt.tight_layout()
    if counter == 0:
        print('NO DATA')
        return
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_scatter.png")


def plot_session_summary_dropout_scatter(IDS,version=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of the dropout performance change for each feature against each other feature 
    '''
    directory=pgt.get_directory(version)
    # make figure    
    allW = None
    counter = 0
    first = True
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version, cross_validation_dropout=True)
        except:
            pass
        else:
            if first:
                fig,ax = plt.subplots(nrows=len(session_summary[2])-1,ncols=len(session_summary[2])-1,figsize=(11,10))        
                first = False 
            d = session_summary[2]
            l = session_summary[3]
            dropout = d
            labels = l
            dropout= [d[x] for x in labels[1:]]
            labels = labels[1:]
            for i in np.arange(0,np.shape(dropout)[0]):
                if i < np.shape(dropout)[0]-1:
                    for j in np.arange(1, i+1):
                        ax[i,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                        ax[i,j-1].set_xticks([])
                        ax[i,j-1].set_yticks([])
                        for spine in ax[i,j-1].spines.values():
                            spine.set_visible(False)
                for j in np.arange(i+1,np.shape(dropout)[0]):
                    ax[i,j-1].axvline(0,color='k',alpha=0.1)
                    ax[i,j-1].axhline(0,color='k',alpha=0.1)
                    ax[i,j-1].plot(dropout[j], dropout[i],'o',alpha=0.5)
                    ax[i,j-1].set_xlabel(ps.clean_dropout([labels[j]])[0],fontsize=12)
                    ax[i,j-1].set_ylabel(ps.clean_dropout([labels[i]])[0],fontsize=12)
                    ax[i,j-1].xaxis.set_tick_params(labelsize=12)
                    ax[i,j-1].yaxis.set_tick_params(labelsize=12)
                    if i == 0:
                        ax[i,j-1].set_ylim(-80,5)
            counter+=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"dropout_scatter.png")


def plot_session_summary_weight_avg_scatter(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against each other weight, plotting the average weight for each session
    '''
    directory = pgt.get_directory(version)

    # make figure    
    fig,ax = plt.subplots(nrows=nel,ncols=nel,figsize=(11,10))
    allW = None
    counter = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = ps.get_weights_list(weights)
            for i in np.arange(0,np.shape(W)[0]):
                if i < np.shape(W)[0]-1:
                    for j in np.arange(1, i+1):
                        ax[i,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                        ax[i,j-1].set_xticks([])
                        ax[i,j-1].set_yticks([])
                        for spine in ax[i,j-1].spines.values():
                            spine.set_visible(False)
                for j in np.arange(i+1,np.shape(W)[0]):
                    ax[i,j-1].axvline(0,color='k',alpha=0.1)
                    ax[i,j-1].axhline(0,color='k',alpha=0.1)
                    meanWj = np.mean(W[j,:])
                    meanWi = np.mean(W[i,:])
                    stdWj = np.std(W[j,:])
                    stdWi = np.std(W[i,:])
                    ax[i,j-1].plot([meanWj, meanWj], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                    ax[i,j-1].plot(meanWj+[-stdWj,stdWj], [meanWi, meanWi],'k-',alpha=0.1)
                    ax[i,j-1].plot(meanWj, meanWi,'o',alpha=0.5)
                    ax[i,j-1].set_xlabel(ps.clean_weights([weights_list[j]])[0],fontsize=12)
                    ax[i,j-1].set_ylabel(ps.clean_weights([weights_list[i]])[0],fontsize=12)
                    ax[i,j-1].xaxis.set_tick_params(labelsize=12)
                    ax[i,j-1].yaxis.set_tick_params(labelsize=12)
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter.png")


# UPDATE_REQUIRED
def plot_session_summary_weight_avg_scatter_1_2(IDS,label1='late_task0',label2='timing1D',directory=None,savefig=False,group_label="",nel=3,fs1=12,fs2=12,filetype='.png',plot_error=True):
    '''
        Makes a summary plot of the average weights of task0 against omission weights for each session
        Also computes a regression line, and returns the linear model
    '''
    if type(directory) == type(None):
        directory = global_directory
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(3,4))
    allx = []
    ally = []
    counter = 0
    ax.axvline(0,color='k',alpha=0.5,ls='--')
    ax.axhline(0,color='k',alpha=0.5,ls='--')
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = ps.get_weights_list(weights)
            xdex = np.where(np.array(weights_list) == label1)[0][0]
            ydex = np.where(np.array(weights_list) == label2)[0][0]

            meanWj = np.mean(W[xdex,:])
            meanWi = np.mean(W[ydex,:])
            allx.append(meanWj)
            ally.append(meanWi)
            stdWj = np.std(W[xdex,:])
            stdWi = np.std(W[ydex,:])
            if plot_error:
                ax.plot([meanWj, meanWj], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax.plot(meanWj+[-stdWj,stdWj], [meanWi, meanWi],'k-',alpha=0.1)
            ax.plot(meanWj, meanWi,'ko',alpha=0.5)
            ax.set_xlabel(ps.clean_weights([weights_list[xdex]])[0],fontsize=fs1)
            ax.set_ylabel(ps.clean_weights([weights_list[ydex]])[0],fontsize=fs1)
            ax.xaxis.set_tick_params(labelsize=fs2)
            ax.yaxis.set_tick_params(labelsize=fs2)
            counter+=1
    if counter == 0:
        print('NO DATA')
        return
    x = np.array(allx).reshape((-1,1))
    y = np.array(ally)
    model = LinearRegression(fit_intercept=True).fit(x,y)
    sortx = np.sort(allx).reshape((-1,1))
    y_pred = model.predict(sortx)
    ax.plot(sortx,y_pred, 'r--')
    score = round(model.score(x,y),2)
    #plt.text(sortx[0],y_pred[-1],"Omissions = "+str(round(model.coef_[0],2))+"*Task \nr^2 = "+str(score),color="r",fontsize=fs2)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_"+label1+"_"+label2+filetype)
    return model


def plot_session_summary_weight_avg_scatter_task0(IDS,version=None,savefig=False,group_label="",nel=3,fs1=12,fs2=12,filetype='.png',plot_error=True):
    '''
        Makes a summary plot of the average weights of task0 against omission weights for each session
        Also computes a regression line, and returns the linear model
    '''
    directory=pgt.get_directory(version) 
    style=pstyle.get_style()
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(3.75,5))
    allx = []
    ally = []
    counter = 0
    ax.axvline(0,color='k',alpha=0.5,ls='--')
    ax.axhline(0,color='k',alpha=0.5,ls='--')
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = ps.get_weights_list(weights)
            xdex = np.where(np.array(weights_list) == 'task0')[0][0]
            ydex = np.where(np.array(weights_list) == 'omissions1')[0][0]

            meanWj = np.mean(W[xdex,:])
            meanWi = np.mean(W[ydex,:])
            allx.append(meanWj)
            ally.append(meanWi)
            stdWj = np.std(W[xdex,:])
            stdWi = np.std(W[ydex,:])
            if plot_error:
                ax.plot([meanWj, meanWj], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax.plot(meanWj+[-stdWj,stdWj], [meanWi, meanWi],'k-',alpha=0.1)
            ax.plot(meanWj, meanWi,'ko',alpha=0.5)
            ax.set_xlabel('Avg. '+ps.clean_weights([weights_list[xdex]])[0]+' weight',fontsize=style['label_fontsize'])
            ax.set_ylabel('Avg. '+ps.clean_weights([weights_list[ydex]])[0]+' weight',fontsize=style['label_fontsize'])
            ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
            ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
            counter+=1
    if counter == 0:
        print('NO DATA')
        return
    x = np.array(allx).reshape((-1,1))
    y = np.array(ally)
    model = LinearRegression(fit_intercept=False).fit(x,y)
    sortx = np.sort(allx).reshape((-1,1))
    y_pred = model.predict(sortx)
    ax.plot(sortx,y_pred, 'r--')
    score = round(model.score(x,y),2)
    #plt.text(sortx[0],y_pred[-1],"Omissions = "+str(round(model.coef_[0],2))+"*Task \nr^2 = "+str(score),color="r",fontsize=fs2)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_task0"+filetype)
    return model


def plot_session_summary_weight_avg_scatter_hits(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against the total number of hits
    '''
    directory=pgt.get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=2,ncols=nel+1,figsize=(14,6))
    allW = None
    counter = 0
    xmax = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            fit = session_summary[7]
            hits = np.sum(fit['psydata']['hits'])
            xmax = np.max([hits, xmax])
            weights  = session_summary[1]
            weights_list = ps.get_weights_list(weights)
            for i in np.arange(0,np.shape(W)[0]):
                ax[0,i].axhline(0,color='k',alpha=0.1)
                meanWi = np.mean(W[i,:])
                stdWi = np.std(W[i,:])
                ax[0,i].plot([hits, hits], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax[0,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[0,i].set_xlabel('hits',fontsize=12)
                ax[0,i].set_ylabel(ps.clean_weights([weights_list[i]])[0],fontsize=12)
                ax[0,i].xaxis.set_tick_params(labelsize=12)
                ax[0,i].yaxis.set_tick_params(labelsize=12)
                ax[0,i].set_xlim(xmin=0,xmax=xmax)

                meanWi = transform(np.mean(W[i,:]))
                stdWiPlus = transform(np.mean(W[i,:])+np.std(W[i,:]))
                stdWiMinus =transform(np.mean(W[i,:])-np.std(W[i,:])) 
                ax[1,i].plot([hits, hits], [stdWiMinus, stdWiPlus],'k-',alpha=0.1)
                ax[1,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[1,i].set_xlabel('hits',fontsize=12)
                ax[1,i].set_ylabel(ps.clean_weights([weights_list[i]])[0],fontsize=12)
                ax[1,i].xaxis.set_tick_params(labelsize=12)
                ax[1,i].yaxis.set_tick_params(labelsize=12)
                ax[1,i].set_xlim(xmin=0,xmax=xmax)
                ax[1,i].set_ylim(ymin=0,ymax=1)

            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_hits.png")


def plot_session_summary_weight_avg_scatter_false_alarms(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against the total number of false_alarms
    '''
    directory = pgt.get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=2,ncols=nel+1,figsize=(14,6))
    allW = None
    counter = 0
    xmax = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            fit = session_summary[7]
            hits = np.sum(fit['psydata']['false_alarms'])
            xmax = np.max([hits, xmax])
            weights  = session_summary[1]
            weights_list = ps.clean_weights(ps.get_weights_list(weights))
            for i in np.arange(0,np.shape(W)[0]):
                ax[0,i].axhline(0,color='k',alpha=0.1)
                meanWi = np.mean(W[i,:])
                stdWi = np.std(W[i,:])
                ax[0,i].plot([hits, hits], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax[0,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[0,i].set_xlabel('false_alarms',fontsize=12)
                ax[0,i].set_ylabel(weights_list[i],fontsize=12)
                ax[0,i].xaxis.set_tick_params(labelsize=12)
                ax[0,i].yaxis.set_tick_params(labelsize=12)
                ax[0,i].set_xlim(xmin=0,xmax=xmax)

                meanWi = transform(np.mean(W[i,:]))
                stdWiPlus = transform(np.mean(W[i,:])+np.std(W[i,:]))
                stdWiMinus =transform(np.mean(W[i,:])-np.std(W[i,:])) 
                ax[1,i].plot([hits, hits], [stdWiMinus, stdWiPlus],'k-',alpha=0.1)
                ax[1,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[1,i].set_xlabel('false_alarms',fontsize=12)
                ax[1,i].set_ylabel(weights_list[i],fontsize=12)
                ax[1,i].xaxis.set_tick_params(labelsize=12)
                ax[1,i].yaxis.set_tick_params(labelsize=12)
                ax[1,i].set_xlim(xmin=0,xmax=xmax)
                ax[1,i].set_ylim(ymin=0,ymax=1)

            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_false_alarms.png")


def plot_session_summary_weight_avg_scatter_miss(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a scatter plot of each weight against the total number of miss
    '''
    directory=pgt.get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=2,ncols=nel+1,figsize=(14,6))
    allW = None
    counter = 0
    xmax = 0
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            fit = session_summary[7]
            hits = np.sum(fit['psydata']['misses'])
            xmax = np.max([hits, xmax])
            weights  = session_summary[1]
            weights_list = ps.clean_weights(ps.get_weights_list(weights))
            for i in np.arange(0,np.shape(W)[0]):
                ax[0,i].axhline(0,color='k',alpha=0.1)
                meanWi = np.mean(W[i,:])
                stdWi = np.std(W[i,:])
                ax[0,i].plot([hits, hits], meanWi+[-stdWi, stdWi],'k-',alpha=0.1)
                ax[0,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[0,i].set_xlabel('misses',fontsize=12)
                ax[0,i].set_ylabel(weights_list[i],fontsize=12)
                ax[0,i].xaxis.set_tick_params(labelsize=12)
                ax[0,i].yaxis.set_tick_params(labelsize=12)
                ax[0,i].set_xlim(xmin=0,xmax=xmax)

                meanWi = transform(np.mean(W[i,:]))
                stdWiPlus = transform(np.mean(W[i,:])+np.std(W[i,:]))
                stdWiMinus =transform(np.mean(W[i,:])-np.std(W[i,:])) 
                ax[1,i].plot([hits, hits], [stdWiMinus, stdWiPlus],'k-',alpha=0.1)
                ax[1,i].plot(hits, meanWi,'o',alpha=0.5)
                ax[1,i].set_xlabel('misses',fontsize=12)
                ax[1,i].set_ylabel(weights_list[i],fontsize=12)
                ax[1,i].xaxis.set_tick_params(labelsize=12)
                ax[1,i].yaxis.set_tick_params(labelsize=12)
                ax[1,i].set_xlim(xmin=0,xmax=xmax)
                ax[1,i].set_ylim(ymin=0,ymax=1)

            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_avg_scatter_misses.png")


def plot_session_summary_weight_trajectory(IDS,version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a summary plot by plotting each weights trajectory across each session. Plots the average trajectory in bold
        this function is super hacky. average is wrong, and doesnt properly align time due to consumption bouts. But gets the general pictures. 
    '''
    directory= pgt.get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=nel+1,ncols=1,figsize=(6,10))
    allW = []
    counter = 0
    xmax  =  []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            W = session_summary[6]
            weights  = session_summary[1]
            weights_list = ps.clean_weights(ps.get_weights_list(weights))
            for i in np.arange(0,np.shape(W)[0]):
                ax[i].plot(W[i,:],alpha = 0.2)
                ax[i].set_ylabel(weights_list[i],fontsize=12)

                xmax.append(len(W[i,:]))
                ax[i].set_xlim(0,np.max(xmax))
                ax[i].xaxis.set_tick_params(labelsize=12)
                ax[i].yaxis.set_tick_params(labelsize=12)
                if i == np.shape(W)[0] -1:
                    ax[i].set_xlabel('Flash #',fontsize=12)
            W = np.pad(W,([0,0],[0,4000]),'constant',constant_values=0)
            allW.append(W[:,0:4000])
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    allW = np.mean(np.array(allW),0)
    for i in np.arange(0,np.shape(W)[0]):
        ax[i].axhline(0, color='k')
        ax[i].plot(allW[i,:],'k',alpha = 1,lw=3)
        if i> 0:
            ax[i].set_ylim(ymin=-2.5)
        ax[i].set_xlim(0,4000)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_trajectory.png")


def plot_session_summary_logodds(IDS,version=None,savefig=False,group_label="",cross_validation=True,hit_threshold=0):
    '''
        Makes a summary plot of the log-odds of the model fits = log(prob(lick|lick happened)/prob(lick|no lick happened))
    '''
    directory=pgt.get_directory(version)
    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(10,4.5))
    logodds=[]
    counter =0
    ids= []
    for id in IDS:
        try:
            #session_summary = get_session_summary(id)
            filenamed = directory + str(id) + ".pkl" 
            output = load(filenamed)
            if type(output) is not dict:
                labels = ['models', 'labels', 'boots', 'hyp', 'evd', 'wMode', 'hess', 'credibleInt', 'weights', 'ypred','psydata','cross_results','cv_pred','metadata']
                fit = dict((x,y) for x,y in zip(labels, output))
            else:
                fit = output
            if np.sum(fit['psydata']['hits']) < hit_threshold:
                raise Exception('below hit threshold')
        except:
            pass
        else:
            if cross_validation:
                lickedp = np.mean(fit['cv_pred'][fit['psydata']['y'] ==2])
                nolickp = np.mean(fit['cv_pred'][fit['psydata']['y'] ==1])
            else:
                lickedp = np.mean(fit['ypred'][fit['psydata']['y'] ==2])
                nolickp = np.mean(fit['ypred'][fit['psydata']['y'] ==1])
            ax[0].plot(nolickp,lickedp, 'o', alpha = 0.5)
            logodds.append(np.log(lickedp/nolickp))
            ids.append(id)
            counter +=1
    if counter == 0:
        print('NO DATA')
        return
    ax[0].set_ylabel('P(lick|lick)', fontsize=12)
    ax[0].set_xlabel('P(lick|no-lick)', fontsize=12)
    ax[0].plot([0,1],[0,1], 'k--',alpha=0.2)
    ax[0].xaxis.set_tick_params(labelsize=12)
    ax[0].yaxis.set_tick_params(labelsize=12)
    ax[0].set_ylim(0,1)
    ax[0].set_xlim(0,1)
    ax[1].hist(np.array(logodds),bins=30)
    ax[1].set_ylabel('Count', fontsize=12)
    ax[1].set_xlabel('Log-Odds', fontsize=12)
    ax[1].xaxis.set_tick_params(labelsize=12)
    ax[1].yaxis.set_tick_params(labelsize=12)
    meanscore = np.median(np.array(logodds))
    ax[1].plot(meanscore, ax[1].get_ylim()[1],'rv')
    ax[1].axvline(meanscore,color='r', alpha=0.3)


    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"weight_logodds.png")

    median = np.argsort(np.array(logodds))[len(logodds)//2]
    best = np.argmax(np.array(logodds))
    worst = np.argmin(np.array(logodds)) 
    print("Log-Odds Summary:")
    print('Worst  Session: ' + str(ids[worst]) + " " + str(logodds[worst]))
    print('Median Session: ' + str(ids[median]) + " " + str(logodds[median]))
    print('Best   Session: ' + str(ids[best]) + " " + str(logodds[best]))      

# UPDATE_REQUIRED
def get_all_weights(IDS,directory=None):
    '''
        Returns a concatenation of all weights for every session in IDS
    '''
    if type(directory) == type(None):
        directory = global_directory
    weights = None
    for id in IDS:
        try:
            session_summary = get_session_summary(id,directory=directory)
        except:
            pass
        else:
            if weights is None:
                weights = session_summary[6]
            else:
                weights = np.concatenate([weights, session_summary[6]],1)
    return weights



def plot_session_summary_roc(IDS,version=None,savefig=False,group_label="",verbose=True,cross_validation=True,fs1=12,fs2=12,filetype=".png"):
    '''
        Make a summary plot of the histogram of AU.ROC values for all sessions in IDS.
    '''
    directory=pgt.get_directory(version)
    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    scores = []
    ids = []
    counter = 0
    hits = []
    for id in IDS:
        try:
            session_summary = get_session_summary(id,version=version)
        except:
            pass
        else:
            fit = session_summary[7]
            roc = ps.compute_model_roc(fit,plot_this=False,cross_validation=cross_validation)
            scores.append(roc)
            ids.append(id)
            hits.append(np.sum(fit['psydata']['hits']))
            counter +=1

    if counter == 0:
        print('NO DATA')
        return
    ax.set_xlim(0.5,1)
    ax.hist(np.array(scores),bins=25)
    ax.set_ylabel('Count', fontsize=fs1)
    ax.set_xlabel('ROC-AUC', fontsize=fs1)
    ax.xaxis.set_tick_params(labelsize=fs2)
    ax.yaxis.set_tick_params(labelsize=fs2)
    meanscore = np.median(np.array(scores))
    ax.plot(meanscore, ax.get_ylim()[1],'rv')
    ax.axvline(meanscore,color='r', alpha=0.3)
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"roc"+filetype)
    if verbose:
        median = np.argsort(np.array(scores))[len(scores)//2]
        best = np.argmax(np.array(scores))
        worst = np.argmin(np.array(scores)) 
        print("ROC Summary:")
        print('Worst  Session: ' + str(ids[worst]) + " " + str(scores[worst]))
        print('Median Session: ' + str(ids[median]) + " " + str(scores[median]))
        print('Best   Session: ' + str(ids[best]) + " " + str(scores[best]))     

    plt.figure()
    plt.plot(scores, hits, 'ko')
    plt.xlim(0.5,1)
    plt.ylim(0,200)
    plt.ylabel('Hits',fontsize=12)
    plt.xlabel('ROC-AUC',fontsize=12)
    plt.gca().xaxis.set_tick_params(labelsize=12)
    plt.gca().yaxis.set_tick_params(labelsize=12)    
    plt.tight_layout()
    if savefig:
        plt.savefig(directory+"figures_summary/summary_"+group_label+"roc_vs_hits"+filetype)
    return scores, ids 



