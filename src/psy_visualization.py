import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV as logregcv
from sklearn.linear_model import LogisticRegression as logreg

import psy_tools as ps
import psy_style as pstyle
import psy_timing_tools as pt
import psy_metrics_tools as pm
import psy_general_tools as pgt

 
def get_strategy_list(version):
    '''
        Returns a sorted list of the strategies in model <version>

        Raises an exception if the model version is not recognized. 
    '''
    if version in [20]:
        strategies=['bias','omissions','omissions1','task0','timing1D']
    else:
        raise Exception('Unknown model version')
    return strategies


def plot_session_summary(summary_df,version=None,savefig=False,group_label=""):
    '''
        Makes a series of summary plots for all the sessions in summary_df
        group_label (str) saves model figures with the label. Does not do any filtering on summary_df.  
    '''
    plot_session_summary_priors(summary_df,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout(summary_df,version=version,cross_validation=False,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout(summary_df,version=version,cross_validation=True,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_dropout_scatter(IDS, version=version, savefig=savefig, group_label=group_label); plt.close('all')
    plot_session_summary_weights(summary_df,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_weight_range(summary_df,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_weight_avg_scatter(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_session_summary_weight_avg_scatter_task0(summary_df,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    
    # Plot session-wise metrics against strategy weights
    event=['hits','fa','cr','miss','aborts','lick_hit_fraction','lick_fraction','trial_hit_fraction','fraction_engaged']
    for e in event:
        plot_session_summary_weight_avg_scatter_task_events(summary_df,e,version=version,savefig=savefig,group_label=group_label); plt.close('all')

    # Plot image-wise metrics, averaged across sessions
    event = ['omissions1','task0','timing1D','omissions','bias',
        'miss', 'reward_rate','change','FA','CR','lick_bout_rate','RT',
        'engaged','hit','lick_hit_fraction_rate']
    for e in event:
        plot_session_summary_trajectory(summary_df,e,version=version,savefig=savefig,group_label=group_label); plt.close('all')

    plot_session_summary_roc(summary_df,version=version,savefig=savefig,group_label=group_label); plt.close('all')
    plot_static_comparison(IDS,version=version,savefig=savefig,group_label=group_label); plt.close('all')


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

    # TODO, Issue #175    
    print('WARNING!!!!')
    print('cross_validation=True/False has not been validated during re-build') 
 
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
            filename=directory+"summary_"+group_label+"dropout_cv"+filetype
            plt.savefig(filename)
            print('Figured saved to: '+filename)
        else:
            filename=directory+"summary_"+group_label+"dropout"+filetype
            plt.savefig(filename)
            print('Figured saved to: '+filename)


def plot_session_summary_weights(summary_df,version=None, savefig=False,group_label="",fs1=12,fs2=12,filetype='.svg'):
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
        filename=directory+"summary_"+group_label+"weights"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)


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
        filename=directory+"summary_"+group_label+"weight_range.png"
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_dropout_scatter(summary_df,version=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of the dropout performance change for each feature against each other feature 
    '''

    # Make Figure
    strategies = get_strategy_list(version)
    fig,ax = plt.subplots(nrows=len(strategies)-1,ncols=len(strategies)-1,figsize=(11,10))        

    for index, strat in enumerate(strategies):
        if index < len(strategies)-1:
            for j in np.arange(1, index+1):
                ax[index,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                ax[index,j-1].set_xticks([])
                ax[index,j-1].set_yticks([])
                for spine in ax[index,j-1].spines.values():
                    spine.set_visible(False)
        for j in np.arange(index+1,len(strategies)):
            ax[index,j-1].axvline(0,color='k',linestyle='--',alpha=0.5)
            ax[index,j-1].axhline(0,color='k',linestyle='--',alpha=0.5)
            ax[index,j-1].plot(summary_df['dropout_'+strategies[j]],summary_df['dropout_'+strat],'o',alpha=0.5)
            ax[index,j-1].set_xlabel(ps.clean_dropout([strategies[j]])[0],fontsize=12)
            ax[index,j-1].set_ylabel(ps.clean_dropout([strat])[0],fontsize=12)
            ax[index,j-1].xaxis.set_tick_params(labelsize=12)
            ax[index,j-1].yaxis.set_tick_params(labelsize=12)
            if index == 0:
                ax[index,j-1].set_ylim(-80,5)

    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename=directory+"summary_"+group_label+"dropout_scatter.png"
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_weight_avg_scatter(summary_df,version=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of each weight against each other weight, plotting the average weight for each session
    '''
    # make figure    
    strategies = get_strategy_list(version)
    style=pstyle.get_style()
    fig,ax = plt.subplots(nrows=len(strategies)-1,ncols=len(strategies)-1,figsize=(11,10))

    for i in np.arange(0,len(strategies)-1):
        if i < len(strategies)-1:
            for j in np.arange(1, i+1):
                ax[i,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                ax[i,j-1].set_xticks([])
                ax[i,j-1].set_yticks([])
                for spine in ax[i,j-1].spines.values():
                    spine.set_visible(False)
        for j in np.arange(i+1,len(strategies)):
            ax[i,j-1].axvline(0,color=style['axline_color'],alpha=style['axline_alpha'],linestyle=style['axline_linestyle'])
            ax[i,j-1].axhline(0,color=style['axline_color'],alpha=style['axline_alpha'],linestyle=style['axline_linestyle'])
            ax[i,j-1].plot(summary_df['avg_weight_'+strategies[j]],summary_df['avg_weight_'+strategies[i]],
                'o',alpha=style['data_alpha'],color=style['data_color_all'])
            ax[i,j-1].set_xlabel(ps.clean_weights([strategies[j]])[0],fontsize=style['label_fontsize_dense'])
            ax[i,j-1].set_ylabel(ps.clean_weights([strategies[i]])[0],fontsize=style['label_fontsize_dense'])
            ax[i,j-1].xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
            ax[i,j-1].yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])

    plt.tight_layout()
    if savefig:
        directory = pgt.get_directory(version,subdirectory='figures')
        filename=directory+"summary_"+group_label+"weight_avg_scatter.png"
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_weight_avg_scatter_task0(summary_df, version=None,savefig=False,group_label="",filetype='.png',plot_error=True):
    '''
        Makes a summary plot of the average weights of task0 against omission weights for each session
        Also computes a regression line, and returns the linear model
    '''

    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(3.75,5))  
    strategies = get_strategy_list(version)
    style = pstyle.get_style()
    plt.plot(summary_df['avg_weight_task0'],summary_df['avg_weight_omissions1'],'ko',alpha=style['data_alpha'])
    ax.set_xlabel('Avg. '+ps.clean_weights(['task0'])[0]+' weight',fontsize=style['label_fontsize'])
    ax.set_ylabel('Avg. '+ps.clean_weights(['omissions1'])[0]+' weight',fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.axvline(0,color=style['axline_color'],
        alpha=style['axline_alpha'],
        ls=style['axline_linestyle'])
    ax.axhline(0,color=style['axline_color'],
        alpha=style['axline_alpha'],
        ls=style['axline_linestyle'])

    # Compute Linear Regression
    x = np.array(summary_df['avg_weight_task0'].values).reshape((-1,1))
    y = np.array(summary_df['avg_weight_omissions1'].values)
    model = LinearRegression(fit_intercept=False).fit(x,y)
    sortx = np.sort(x)
    y_pred = model.predict(sortx)
    ax.plot(sortx,y_pred, 
        color=style['regression_color'], 
        linestyle=style['regression_linestyle'])
    score = round(model.score(x,y),2)

    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures') 
        filename=directory+"summary_"+group_label+"weight_avg_scatter_task0"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)
    return model


def plot_session_summary_weight_avg_scatter_task_events(summary_df,event,version=None,savefig=False,group_label=""):
    '''
        Makes a scatter plot of each weight against the total number of <event>
        <event> needs to be a session-wise metric

        Raises an exception if event is not a session-wise metric
    '''
    
    # Check if we have a discrete session wise event
    if event in ['hits','fa','cr','miss','aborts']:
        df_event = 'num_'+event
    elif event in ['lick_hit_fraction','lick_fraction','trial_hit_fraction','fraction_engaged']:
        df_event = event
    else:
        raise Exception('Bad event type')
    
    # make figure   
    strategies = get_strategy_list(version) 
    style = pstyle.get_style()
    fig,ax = plt.subplots(nrows=1,ncols=len(strategies),figsize=(14,3))
    num_sessions = len(summary_df)
    for index, strat in enumerate(strategies):
        ax[index].plot(summary_df[df_event], summary_df['avg_weight_'+strat].values,'o',alpha=style['data_alpha'],color=style['data_color_all'])
        ax[index].set_xlabel(event,fontsize=style['label_fontsize'])
        ax[index].set_ylabel(ps.clean_weights([strat])[0],fontsize=style['label_fontsize'])
        ax[index].xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
        ax[index].yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
        ax[index].axhline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])

    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename=directory+"summary_"+group_label+"weight_avg_scatter_"+event+".png"
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_trajectory(summary_df,trajectory, version=None,savefig=False,group_label=""):
    '''
        Makes a summary plot by plotting the average value of trajectory over the session
        trajectory needs to be a image-wise metric, with 4800 values for each session.

        Raises an exception if trajectory is not on the approved list.  
    '''

    # Check if we have an image wise metric
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
    style = pstyle.get_style() 
    values = np.vstack(summary_df[plot_trajectory].values)
    mean_values = np.nanmean(values, axis=0)
    std_values = np.nanstd(values, axis=0)
    ax.plot(mean_values,color=style['data_color_all'])
    ax.fill_between(range(0,np.size(values,1)), mean_values-std_values, mean_values+std_values,color=style['data_uncertainty_color'],alpha=style['data_uncertainty_alpha'])
    ax.set_xlim(0,4800)
    ax.axhline(0, color=style['axline_color'],
        linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])
    ax.set_ylabel(ps.clean_weights([trajectory])[0],fontsize=style['label_fontsize']) 
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.set_xlabel('Image #',fontsize=style['label_fontsize'])

    # remove extra axis
    plt.tight_layout()
    
    # Save Figure
    if savefig:
        directory= pgt.get_directory(version,subdirectory='figures')
        filename=directory+"summary_"+group_label+"trajectory_"+trajectory+".png"
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_roc(summary_df,version=None,savefig=False,group_label="",cross_validation=True,filetype=".png"):
    '''
        Make a summary plot of the histogram of AU.ROC values for all sessions 
    '''

    # TODO, Issue #175    
    print('WARNING!!!!')
    print('cross_validation=True/False has not been validated during re-build') 

    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    ax.set_xlim(0.5,1)
    ax.hist(summary_df['session_roc'],bins=25)
    ax.set_ylabel('Count', fontsize=style['label_fontsize'])
    ax.set_xlabel('ROC-AUC', fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    meanscore =summary_df['session_roc'].median()
    ax.plot(meanscore, ax.get_ylim()[1],'rv')
    ax.axvline(meanscore,color=style['regression_color'], 
        linestyle=style['regression_linestyle'],
        alpha=0.75)
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename=directory+"summary_"+group_label+"roc"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)

    best = summary_df['session_roc'].idxmax()
    worst = summary_df['session_roc'].idxmin()
    print("ROC Summary:")
    print('Avg ROC Score : ' +str(np.round(meanscore,3)))
    print('Worst Session : ' + str(summary_df['behavior_session_id'].loc[worst]) + 
        " " + str(np.round(summary_df['session_roc'].loc[worst],3)))
    print('Best Session  : ' + str(summary_df['behavior_session_id'].loc[best]) + 
        " " + str(np.round(summary_df['session_roc'].loc[best],3)))


def plot_static_comparison(summary_df, version=None,savefig=False,group_label=""):
    '''
        Top Level function for comparing static and dynamic logistic regression using ROC scores
    
        Computes the values with :
            get_all_static_roc
            get_static_roc
            get_static_design_matrix
        plots with:
            plot_static_comparison_inner
             
    '''
    summary_df = get_all_static_roc(summary_df, version)
    plot_static_comparison_inner(summary_df,version=version, savefig=savefig, group_label=group_label)


def plot_static_comparison_inner(summary_df,version=None, savefig=False,group_label="",filetype='.png'): 
    '''
        Plots static and dynamic ROC comparisons

        Called by plot_static_comparison
    
    '''
    fig,ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    plt.plot(summary_df['static_session_roc'],summary_df['session_roc'],'o',color=style['data_color_all'],alpha=style['data_alpha'])
    plt.plot([0.5,1],[0.5,1],color=style['axline_color'],
        alpha=style['axline_alpha'], linestyle=style['axline_linestyle'])
    plt.ylabel('Dynamic ROC',fontsize=style['label_fontsize'])
    plt.xlabel('Static ROC',fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        plt.savefig(directory+"summary_static_comparison"+group_label+filetype)


def get_all_static_roc(summary_df, version):
    '''
        Iterates through sessions and gets static ROC scores
        Used by plot_static_comparison

        returns the summary_df with an added column "static_session_roc".
            values are the static au.ROC, or NaN 
    '''
    summary_df = summary_df.set_index('behavior_session_id')
    for index, bsid in enumerate(tqdm(summary_df.index.values)):
        try:
            fit = ps.load_fit(bsid, version=version)
            static = get_static_roc(fit)
            summary_df.at[bsid,'static_session_roc'] = static
        except:
            summary_df.at[bsid,'static_session_roc'] = np.nan

    summary_df = summary_df.reset_index()
    return summary_df


def get_static_design_matrix(fit):
    '''
        Returns the design matrix to be used for static logistic regression.
        Does not include bias, because that is added by logreg
        Used by plot_static_comparison
    '''
    X = []
    for index, w in enumerate(fit['weights'].keys()):
        if fit['weights'][w]:
            if not (w=='bias'):
                X.append(fit['psydata']['inputs'][w]) 
    return np.hstack(X)


def get_static_roc(fit,use_cv=False):
    '''
        Returns the area under the ROC curve for a static logistic regression model
        Used by plot_static_comparison
    '''
    X = get_static_design_matrix(fit)
    y = fit['psydata']['y'] - 1
    if use_cv:
        clf = logregcv(cv=10)
    else:
        clf = logreg(penalty='none',solver='lbfgs')
    clf.fit(X,y)
    ypred = clf.predict(X)
    fpr, tpr, thresholds = metrics.roc_curve(y,ypred)
    static_roc = metrics.auc(fpr,tpr)

    return static_roc



