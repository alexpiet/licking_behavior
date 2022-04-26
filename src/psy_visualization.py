import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import psy_style as pstyle
import psy_timing_tools as pt
import psy_metrics_tools as pm
import psy_tools as ps
import psy_general_tools as pgt

# TODO, move ROC vs hist to scatter by manifest
# TODO, figure out CV thing
# TODO, random colors?
# TODO, NaN weights? Check to see what is happening, add to list of things to QC in building the summary table
# TODO, Make a more general "clean_str" function that removes _ and capitalizes, etc
# TODO, make more organized lists of session-wise metrics, and image-wise metrics
# TODO, put elements of summary table into table in more logical order
# TODO, remove nel from these function calls
# TODO, should static comparison be part of main fit

def get_strategy_list(version):
    strategies=['bias','omissions','omissions1','task0','timing1D']
    return strategies

def plot_session_summary(summary_df,version=None,savefig=False,group_label="",nel=4):
    '''
        Makes a series of summary plots for all the IDS
    '''
    plot_session_summary_priors(summary_df,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout(summary_df,version=version,cross_validation=False,savefig=savefig,group_label=group_label); plt.close('all')
    #plot_session_summary_dropout(summary_df,version=version,cross_validation=True,savefig=savefig,group_label=group_label); plt.close('all')
    #plot_session_summary_dropout_scatter(IDS, version=version, savefig=savefig, group_label=group_label); plt.close('all')
    plot_session_summary_weights(summary_df,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_weight_range(summary_df,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    #plot_session_summary_weight_scatter(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    #plot_session_summary_weight_avg_scatter(IDS,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    plot_session_summary_weight_avg_scatter_task0(summary_df,version=version,savefig=savefig,group_label=group_label,nel=nel); plt.close('all')
    
    # Plot session-wise metrics against strategy weights
    event=['hits','fa','cr','miss','aborts','lick_hit_fraction','lick_fraction','trial_hit_fraction','fraction_engaged']
    for e in event:
        plot_session_summary_weight_avg_scatter_task_event(summary_df,e,version=version,savefig=savefig,group_label=group_label); plt.close('all')

    # Plot image-wise metrics, averaged across sessions
    event = ['omissions1','task0','timing1D','omissions','bias',
        'miss', 'reward_rate','change','FA','CR','lick_bout_rate','RT',
        'engaged','hit','lick_hit_fraction_rate']
    for e in event:
        plot_session_summary_trajectory(summary_df,e,version=version,savefig=savefig,group_label=group_label); plt.close('all')

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



def plot_session_summary_priors(summary_df,version=None,savefig=False,group_label="",fs1=12,fs2=12,filetype='.png'):
    '''
        Make a summary plot of the priors on each feature
    '''

    # plot data
    fig,ax = plt.subplots(figsize=(4,6))
    strategies = get_strategy_list(version) 
    num_sessions = len(summary_df)
    for index, strat in enumerate(strategies):
        ax.plot([index]*num_sessions, summary_df['prior_'+strat].values,'o',alpha=0.5)
        strat_mean = summary_df['prior_'+strat].mean()
        ax.plot([index-.25,index+.25], [strat_mean, strat_mean], 'k-',lw=3)
        if np.mod(index, 2) == 0:
            plt.axvspan(index-.5,index+.5, color='k',alpha=0.1)

    # Clean up
    plt.ylabel('Smoothing Prior, $\sigma$\n <-- smooth           variable --> ',fontsize=fs1)
    plt.yscale('log')
    plt.ylim(0.0001, 20)  
    ax.set_xticks(np.arange(0,len(strategies)))
    weights_list = ps.clean_weights(strategies)
    ax.set_xticklabels(weights_list,fontsize=fs2,rotation=90)
    ax.axhline(0.001,color='k',alpha=0.2)
    ax.axhline(0.01,color='k',alpha=0.2)
    ax.axhline(0.1,color='k',alpha=0.2)
    ax.axhline(1,color='k',alpha=0.2)
    ax.axhline(10,color='k',alpha=0.2)
    plt.yticks(fontsize=fs2-4,rotation=90)
    ax.xaxis.tick_top()
    ax.set_xlim(xmin=-.5)
    plt.tight_layout()

    # Save
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename = directory+"summary_"+group_label+"prior"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)

def plot_session_summary_dropout(summary_df,version=None,cross_validation=True,savefig=False,group_label="",model_evidence=False,fs1=12,fs2=12,filetype='.png'):
    '''
        Make a summary plot showing the fractional change in either model evidence (not cross-validated), or log-likelihood (cross-validated)
    '''

 
    # make figure    
    fig,ax = plt.subplots(figsize=(7.2,6))
    strategies = get_strategy_list(version)
    num_sessions = len(summary_df)
    for index, strat in enumerate(strategies):
        ax.plot([index]*num_sessions, summary_df['dropout_'+strat].values,'o',alpha=0.5)
        strat_mean = summary_df['dropout_'+strat].mean()
        ax.plot([index-.25,index+.25], [strat_mean, strat_mean], 'k-',lw=3)
        if np.mod(index,2) == 0:
            plt.axvspan(index-.5,index+.5,color='k', alpha=0.1)

    # Clean up
    ax.axhline(0,color='k',alpha=0.2)
    if cross_validation:
        plt.ylabel('% Change in CV Likelihood \n <-- Worse Fit',fontsize=fs1)
    else:
        plt.ylabel('% Change in Likelihood \n <-- Worse Fit',fontsize=fs1)
    plt.yticks(fontsize=fs2-4,rotation=90) 
    ax.set_xticks(np.arange(0,len(strategies)))
    ax.set_xticklabels(ps.clean_weights(strategies),fontsize=fs2, rotation = 90)
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.xlim(-0.5,len(strategies) - 0.5)
    plt.ylim(-80,5)

    # Save
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        if cross_validation:
            plt.savefig(directory+"summary_"+group_label+"dropout_cv"+filetype)
        else:
            plt.savefig(directory+"summary_"+group_label+"dropout"+filetype)


def plot_session_summary_weights(summary_df,version=None, savefig=False,group_label="",return_weights=False,fs1=12,fs2=12,filetype='.svg',hit_threshold=0):
    '''
        Makes a summary plot showing the average weight value for each session
    '''

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    strategies = get_strategy_list(version)
    num_sessions = len(summary_df)
    for index, strat in enumerate(strategies):
        ax.plot([index]*num_sessions, summary_df['avg_weight_'+strat].values,'o',alpha=0.5)
        strat_mean = summary_df['avg_weight_'+strat].mean()
        ax.plot([index-.25,index+.25], [strat_mean, strat_mean], 'k-',lw=3)
        if np.mod(index,2) == 0:
            plt.axvspan(index-.5,index+.5,color='k', alpha=0.1)

    # Clean up
    ax.set_xticks(np.arange(0,len(strategies)))
    plt.ylabel('Avg. Weights across each session',fontsize=fs1)
    ax.axhline(0,color='k',alpha=0.2)
    ax.set_xticklabels(ps.clean_weights(strategies),fontsize=fs2, rotation = 90)
    ax.xaxis.tick_top()
    plt.yticks(fontsize=fs2-4,rotation=90)
    plt.tight_layout()
    plt.xlim(-0.5,len(strategies) - 0.5)
    
    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        plt.savefig(directory+"summary_"+group_label+"weights"+filetype)


def plot_session_summary_weight_range(summary_df,version=None,savefig=False,group_label=""):
    '''
        Makes a summary plot showing the range of each weight across each session
    '''

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    strategies = get_strategy_list(version)
    num_sessions = len(summary_df)

    for index, strat in enumerate(strategies):
        min_weights = summary_df['weight_'+strat].apply(np.min,axis=0)
        max_weights = summary_df['weight_'+strat].apply(np.max,axis=0)
        range_weights = max_weights-min_weights
        ax.plot([index]*num_sessions, range_weights,'o',alpha=0.5)
        strat_mean = range_weights.mean()
        ax.plot([index-.25,index+.25], [strat_mean, strat_mean], 'k-',lw=3)
        if np.mod(index, 2) == 0:
            plt.axvspan(index-.5,index+.5, color='k',alpha=0.1)

    # Clean up
    ax.set_xticks(np.arange(0,len(strategies)))
    plt.ylabel('Range of Weights across each session',fontsize=12)
    ax.set_xticklabels(ps.clean_weights(strategies),fontsize=12, rotation = 90)
    ax.xaxis.tick_top()
    ax.axhline(0,color='k',alpha=0.2)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.xlim(-0.5,len(strategies) - 0.5)
    
    # Save Figure
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        plt.savefig(directory+"summary_"+group_label+"weight_range.png")


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


def plot_session_summary_weight_avg_scatter_task0(summary_df, version=None,savefig=False,group_label="",nel=3,fs1=12,fs2=12,filetype='.png',plot_error=True):
    '''
        Makes a summary plot of the average weights of task0 against omission weights for each session
        Also computes a regression line, and returns the linear model
    '''


    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(3.75,5))  
    strategies = get_strategy_list(version)
    plt.plot(summary_df['avg_weight_task0'],summary_df['avg_weight_omissions1'],'ko',alpha=0.5)
    style=pstyle.get_style()
    ax.set_xlabel('Avg. '+ps.clean_weights(['task0'])[0]+' weight',fontsize=style['label_fontsize'])
    ax.set_ylabel('Avg. '+ps.clean_weights(['omissions1'])[0]+' weight',fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.axvline(0,color='k',alpha=0.5,ls='--')
    ax.axhline(0,color='k',alpha=0.5,ls='--')

    # Compute Linear Regression
    x = np.array(summary_df['avg_weight_task0'].values).reshape((-1,1))
    y = np.array(summary_df['avg_weight_omissions1'].values)
    model = LinearRegression(fit_intercept=False).fit(x,y)
    sortx = np.sort(x)
    y_pred = model.predict(sortx)
    ax.plot(sortx,y_pred, 'r--')
    score = round(model.score(x,y),2)

    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures') 
        plt.savefig(directory+"summary_"+group_label+"weight_avg_scatter_task0"+filetype)
    return model


def plot_session_summary_weight_avg_scatter_task_events(summary_df,event,version=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of each weight against the total number of <event>
    '''
    if event in ['hits','fa','cr','miss','aborts']:
        df_event = 'num_'+event
    elif event in ['lick_hit_fraction','lick_fraction','trial_hit_fraction','fraction_engaged']:
        df_event = event
    else:
        raise Exception('Bad event type')
    
    # make figure   
    strategies = get_strategy_list(version) 
    fig,ax = plt.subplots(nrows=1,ncols=len(strategies),figsize=(14,3))

    num_sessions = len(summary_df)
    for index, strat in enumerate(strategies):
        ax[index].plot(summary_df[df_event], summary_df['avg_weight_'+strat].values,'o',alpha=0.5)
        ax[index].set_xlabel(event,fontsize=12)
        ax[index].set_ylabel(ps.clean_weights([strat])[0],fontsize=12)
        ax[index].xaxis.set_tick_params(labelsize=12)
        ax[index].yaxis.set_tick_params(labelsize=12)
        ax[index].axhline(0,color='k',linestyle='--',alpha=0.5)

    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        plt.savefig(directory+"summary_"+group_label+"weight_avg_scatter_"+event+".png")


def plot_session_summary_trajectory(summary_df,trajectory, version=None,savefig=False,group_label="",nel=3):
    '''
        Makes a summary plot by plotting each weights trajectory across each session. Plots the average trajectory in bold
    '''

    good_trajectories = ['omissions1','task0','timing1D','omissions','bias',
        'miss', 'reward_rate','change','FA','CR','lick_bout_rate','RT',
        'engaged','hit','lick_hit_fraction_rate']
    if trajectory not in good_trajectories:
        raise Exception('Bad summary variable')
    strategies = get_strategy_list(version)
    if trajectory in strategies:
        plot_trajectory = 'weight_'+trajectory
    else:
        plot_trajectory = trajectory

    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,2.5))  
    values = np.vstack(summary_df[plot_trajectory].values)
    mean_values = np.nanmean(values, axis=0)
    std_values = np.nanstd(values, axis=0)
    ax.plot(mean_values)
    ax.fill_between(range(0,np.size(values,1)), mean_values-std_values, mean_values+std_values,color='k',alpha=.1)
    ax.set_xlim(0,4800)
    ax.axhline(0, color='k',linestyle='--',alpha=0.5)
    ax.set_ylabel(ps.clean_weights([trajectory])[0],fontsize=12) 
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xlabel('Image #',fontsize=12)

    # remove extra axis
    plt.tight_layout()
    
    # Save Figure
    if savefig:
        directory= pgt.get_directory(version,subdirectory='figures')
        plt.savefig(directory+"summary_"+group_label+"trajectory_"+trajectory+".png")


def plot_session_summary_roc(summary_df,version=None,savefig=False,group_label="",verbose=True,cross_validation=True,fs1=12,fs2=12,filetype=".png"):
    '''
        Make a summary plot of the histogram of AU.ROC values for all sessions in IDS.
    '''

    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    ax.set_xlim(0.5,1)
    ax.hist(summary_df['session_roc'],bins=25)
    ax.set_ylabel('Count', fontsize=fs1)
    ax.set_xlabel('ROC-AUC', fontsize=fs1)
    ax.xaxis.set_tick_params(labelsize=fs2)
    ax.yaxis.set_tick_params(labelsize=fs2)
    meanscore =summary_df['session_roc'].median()
    ax.plot(meanscore, ax.get_ylim()[1],'rv')
    ax.axvline(meanscore,color='r', alpha=0.3)
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version)
        plt.savefig(directory+"figures_summary/summary_"+group_label+"roc"+filetype)
    if verbose:
        best = summary_df['session_roc'].idxmax()
        worst = summary_df['session_roc'].idxmin()
        print("ROC Summary:")
        print('Avg ROC Score : ' +str(np.round(meanscore,3)))
        print('Worst Session : ' + str(summary_df['behavior_session_id'].loc[worst]) + 
            " " + str(np.round(summary_df['session_roc'].loc[worst],3)))
        print('Best Session  : ' + str(summary_df['behavior_session_id'].loc[best]) + 
            " " + str(np.round(summary_df['session_roc'].loc[best],3)))






