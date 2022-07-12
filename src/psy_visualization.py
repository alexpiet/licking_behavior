import seaborn as sns
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV as logregcv
from sklearn.linear_model import LogisticRegression as logreg
from mpl_toolkits.axes_grid1 import Divider, Size


import psy_tools as ps
import psy_style as pstyle
import psy_metrics_tools as pm
import psy_general_tools as pgt

def plot_session_summary(summary_df,version=None,savefig=False,group=None):
    '''
        Makes a series of summary plots for all the sessions in summary_df
        group (str) saves model figures with the label. Does not do any filtering  
    '''
    plot_session_summary_priors(summary_df,version=version,savefig=savefig,group=group)
    plot_session_summary_dropout(summary_df,version=version,cross_validation=False,
        savefig=savefig,group=group)
    plot_session_summary_dropout(summary_df,version=version,cross_validation=True,
        savefig=savefig,group=group)
    plot_session_summary_dropout_scatter(summary_df, version=version, savefig=savefig, 
        group=group)
    plot_session_summary_weights(summary_df,version=version,savefig=savefig,group=group)
    plot_session_summary_weight_range(summary_df,version=version,savefig=savefig,
        group=group)
    plot_session_summary_weight_avg_scatter(summary_df,version=version,
        savefig=savefig,group=group)
    plot_session_summary_weight_avg_scatter_task0(summary_df,version=version,
        savefig=savefig,group=group)
    
    # Plot session-wise metrics against strategy weights
    event=['hits','image_false_alarm','image_correct_reject','trial_correct_reject',
        'trial_false_alarm','miss','lick_hit_fraction','lick_fraction',
        'trial_hit_fraction','fraction_engaged']
    for e in event:
        plot_session_summary_weight_avg_scatter_task_events(summary_df,e,
        version=version,savefig=savefig,group=group)

    # Plot image-wise metrics, averaged across sessions
    event = ['omissions1','task0','timing1D','omissions','bias',
        'miss', 'reward_rate','is_change','image_false_alarm','image_correct_reject',
        'lick_bout_rate','RT','engaged','hit','lick_hit_fraction_rate']
    for e in event:
        plot_session_summary_trajectory(summary_df,e,version=version,
            savefig=savefig,group=group)

    plot_session_summary_roc(summary_df,version=version,savefig=savefig,group=group)
    plot_static_comparison(summary_df,version=version,savefig=savefig,group=group)

def plot_all_pivoted_df_by_experience(summary_df, version, 
    experience_type='experience_level', savefig=False, group=None):
    key = ['strategy_dropout_index','strategy_weight_index','lick_hit_fraction',
        'lick_fraction','num_hits']
    flip_key = ['dropout_task0','dropout_timing1D','dropout_omissions1',
        'dropout_omissions']
    for k in key:
        plot_pivoted_df_by_experience(summary_df, k,experience_type=experience_type,
            version=version,flip_index=False,savefig=savefig,group=group)
    for k in flip_key:
        plot_pivoted_df_by_experience(summary_df, k,experience_type=experience_type,
            version=version,flip_index=True,savefig=savefig,group=group)


def plot_all_df_by_experience(summary_df, version, 
    experience_type='experience_level',savefig=False, group=None):
    plot_df_groupby(summary_df,'session_roc',experience_type,hline=0.5,
        version=version,savefig=savefig,group=group)

    key = ['lick_fraction','lick_hit_fraction','trial_hit_fraction',
        'strategy_dropout_index','strategy_weight_index','prior_bias',
        'prior_task0','prior_omissions1','prior_timing1D','avg_weight_bias',
        'avg_weight_task0','avg_weight_omissions1','avg_weight_timing1D']
    for k in key:
        plot_df_groupby(summary_df,k,experience_type,version=version,
            savefig=savefig,group=group)

def plot_all_df_by_cre(summary_df, version,savefig=False, group=None):
    plot_df_groupby(summary_df,'session_roc','cre_line',hline=0.5,
        version=version,savefig=savefig,group=group)

    key = ['lick_fraction','lick_hit_fraction','trial_hit_fraction',
        'strategy_dropout_index','strategy_weight_index','prior_bias',
        'prior_task0','prior_omissions1','prior_timing1D','avg_weight_bias',
        'avg_weight_task0','avg_weight_omissions1','avg_weight_timing1D']
    for k in key:
        plot_df_groupby(summary_df,k,'cre_line',version=version,
            savefig=savefig,group=group)

def plot_strategy_by_cre(summary_df, version=None, savefig=False, group=None):
    '''

    '''
    histogram_df(summary_df, 'strategy_dropout_index',categories='cre_line',
        savefig=savefig, group=group,version=version)
    scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index',
        flip1=True, flip2=True,categories='cre_line',savefig=savefig, group=group, 
        version=version)

## Individual plotting functions below here
################################################################################

def plot_session_summary_priors(summary_df,version=None,savefig=False,group=None,
    filetype='.png'):
    '''
        Make a summary plot of the priors on each feature
    '''

    # plot data
    fig,ax = plt.subplots(figsize=(4,6))
    strategies = pgt.get_strategy_list(version)
    style=pstyle.get_style() 
    num_sessions = len(summary_df)
    for index, strat in enumerate(strategies):
        ax.plot([index]*num_sessions, summary_df['prior_'+strat].values,'o',
        alpha=style['data_alpha'],color=style['data_color_'+strat])
        strat_mean = summary_df['prior_'+strat].mean()
        ax.plot([index-.25,index+.25], [strat_mean, strat_mean], 'k-',lw=3)
        if np.mod(index, 2) == 0:
            plt.axvspan(index-.5,index+.5, color=style['background_color'],
            alpha=style['background_alpha'])

    # Clean up
    plt.ylabel('Smoothing Prior, $\sigma$\n <-- smooth           variable --> ',
        fontsize=style['label_fontsize'])
    plt.yscale('log')
    plt.ylim(0.0001, 20)  
    ax.set_xticks(np.arange(0,len(strategies)))
    weights_list = pgt.get_clean_string(strategies)
    ax.set_xticklabels(weights_list,fontsize=style['axis_ticks_fontsize'],rotation=90)
    ax.axhline(0.001,color=style['axline_color'],alpha=0.2,
        linestyle=style['axline_linestyle'])
    ax.axhline(0.01,color=style['axline_color'],alpha=0.2,
        linestyle=style['axline_linestyle'])
    ax.axhline(0.1,color=style['axline_color'],alpha=0.2,
        linestyle=style['axline_linestyle'])
    ax.axhline(1,color=style['axline_color'],alpha=0.2,
        linestyle=style['axline_linestyle'])
    ax.axhline(10,color=style['axline_color'],alpha=0.2,
        linestyle=style['axline_linestyle'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    ax.set_xlim(xmin=-.5)
    plt.tight_layout()

    # Save
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename = directory+"summary_"+"prior"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_dropout(summary_df,version=None,cross_validation=True,
    savefig=False,group=None,filetype='.png'):
    '''
        Make a summary plot showing the fractional change in either model evidence
        (not cross-validated), or log-likelihood (cross-validated)
    '''

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    strategies = pgt.get_strategy_list(version)
    style = pstyle.get_style()
    num_sessions = len(summary_df)
    if cross_validation:
        dropout_type = 'cv_'
    else:
        dropout_type = ''
    for index, strat in enumerate(strategies):
        ax.plot([index]*num_sessions, summary_df['dropout_'+dropout_type+strat].values,
            'o',alpha=style['data_alpha'],color=style['data_color_'+strat])
        strat_mean = summary_df['dropout_'+dropout_type+strat].mean()
        ax.plot([index-.25,index+.25], [strat_mean, strat_mean], 'k-',lw=3)
        if np.mod(index,2) == 0:
            plt.axvspan(index-.5,index+.5,color=style['background_color'], 
                alpha=style['background_alpha'])

    # Clean up
    ax.axhline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],
        alpha=style['axline_alpha'])
    if cross_validation:
        plt.ylabel('% Change in CV Likelihood \n <-- Worse fit without strategy',
            fontsize=style['label_fontsize'])
    else:
        plt.ylabel('% Change in Model Evidence \n <-- Worse fit without strategy',
            fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize']) 
    ax.set_xticks(np.arange(0,len(strategies)))
    ax.set_xticklabels(pgt.get_clean_string(strategies),
        fontsize=style['axis_ticks_fontsize'], rotation = 90)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.tick_top()
    plt.tight_layout()
    plt.xlim(-0.5,len(strategies) - 0.5)
    plt.ylim(-50,0)

    # Save
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        if cross_validation:
            filename=directory+"summary_"+"dropout_cv"+filetype
            plt.savefig(filename)
            print('Figured saved to: '+filename)
        else:
            filename=directory+"summary_"+"dropout"+filetype
            plt.savefig(filename)
            print('Figured saved to: '+filename)


def plot_session_summary_weights(summary_df,version=None, savefig=False,group=None,
    filetype='.svg'):
    '''
        Makes a summary plot showing the average weight value for each session
    '''

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    strategies = pgt.get_strategy_list(version)
    num_sessions = len(summary_df)
    style = pstyle.get_style()
    for index, strat in enumerate(strategies):
        ax.plot([index]*num_sessions, summary_df['avg_weight_'+strat].values,'o',
            alpha=style['data_alpha'],color=style['data_color_'+strat])
        strat_mean = summary_df['avg_weight_'+strat].mean()
        ax.plot([index-.25,index+.25], [strat_mean, strat_mean], 'k-',lw=3)
        if np.mod(index,2) == 0:
            plt.axvspan(index-.5,index+.5,color=style['background_color'], 
                alpha=style['background_alpha'])

    # Clean up
    ax.set_xticks(np.arange(0,len(strategies)))
    plt.ylabel('Avg. Weights across each session',fontsize=style['label_fontsize'])
    ax.axhline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],
        alpha=style['axline_alpha'])
    ax.set_xticklabels(pgt.get_clean_string(strategies),
        fontsize=style['axis_ticks_fontsize'], rotation = 90)
    ax.xaxis.tick_top()
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.tight_layout()
    plt.xlim(-0.5,len(strategies) - 0.5)
    
    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_"+"weights"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_weight_range(summary_df,version=None,savefig=False,group=None):
    '''
        Makes a summary plot showing the range of each weight across each session
    '''

    # make figure    
    fig,ax = plt.subplots(figsize=(4,6))
    strategies = pgt.get_strategy_list(version)
    style = pstyle.get_style()
    num_sessions = len(summary_df)

    for index, strat in enumerate(strategies):
        min_weights = summary_df['weight_'+strat].apply(np.min,axis=0)
        max_weights = summary_df['weight_'+strat].apply(np.max,axis=0)
        range_weights = max_weights-min_weights
        ax.plot([index]*num_sessions, range_weights,'o',alpha=style['data_alpha'], 
            color=style['data_color_'+strat])
        strat_mean = range_weights.mean()
        ax.plot([index-.25,index+.25], [strat_mean, strat_mean], 'k-',lw=3)
        if np.mod(index, 2) == 0:
            plt.axvspan(index-.5,index+.5, color=style['background_color'],
                alpha=style['background_alpha'])

    # Clean up
    ax.set_xticks(np.arange(0,len(strategies)))
    plt.ylabel('Range of Weights across each session',fontsize=style['label_fontsize'])
    ax.set_xticklabels(pgt.get_clean_string(strategies),
        fontsize=style['axis_ticks_fontsize'], rotation = 90)
    ax.xaxis.tick_top()
    ax.axhline(0,color=style['axline_color'], alpha=style['axline_alpha'], 
        linestyle=style['axline_linestyle'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.tight_layout()
    plt.xlim(-0.5,len(strategies) - 0.5)
    
    # Save Figure
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_"+"weight_range.png"
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_dropout_scatter(summary_df,version=None,savefig=False,
    group=None):
    '''
        Makes a scatter plot of the dropout performance change for each feature
         against each other feature 
    '''

    # Make Figure
    strategies = pgt.get_strategy_list(version)
    fig,ax = plt.subplots(nrows=len(strategies)-1,ncols=len(strategies)-1,
        figsize=(11,10))        
    style = pstyle.get_style()

    for index, strat in enumerate(strategies):
        if index < len(strategies)-1:
            for j in np.arange(1, index+1):
                ax[index,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                ax[index,j-1].set_xticks([])
                ax[index,j-1].set_yticks([])
                for spine in ax[index,j-1].spines.values():
                    spine.set_visible(False)
        for j in np.arange(index+1,len(strategies)):
            ax[index,j-1].axvline(0,color=style['axline_color'],
                linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])
            ax[index,j-1].axhline(0,color=style['axline_color'],
                linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])
            ax[index,j-1].plot(summary_df['dropout_'+strategies[j]],
                summary_df['dropout_'+strat],'o',color=style['data_color_all'],
                alpha=style['data_alpha'])
            ax[index,j-1].set_xlabel(pgt.get_clean_string([strategies[j]])[0],
                fontsize=style['label_fontsize'])
            ax[index,j-1].set_ylabel(pgt.get_clean_string([strat])[0],
                fontsize=style['label_fontsize'])
            ax[index,j-1].xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
            ax[index,j-1].yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
            if index == 0:
                ax[index,j-1].set_ylim(-80,5)

    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_"+"dropout_scatter.png"
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_weight_avg_scatter(summary_df,version=None,savefig=False,
    group=None):
    '''
        Makes a scatter plot of each weight against each other weight, plotting 
        the average weight for each session
    '''
    # make figure    
    strategies = pgt.get_strategy_list(version)
    style=pstyle.get_style()
    fig,ax = plt.subplots(nrows=len(strategies)-1,ncols=len(strategies)-1,
        figsize=(11,10))

    for i in np.arange(0,len(strategies)-1):
        if i < len(strategies)-1:
            for j in np.arange(1, i+1):
                ax[i,j-1].tick_params(top='off',bottom='off', left='off',right='off')
                ax[i,j-1].set_xticks([])
                ax[i,j-1].set_yticks([])
                for spine in ax[i,j-1].spines.values():
                    spine.set_visible(False)
        for j in np.arange(i+1,len(strategies)):
            ax[i,j-1].axvline(0,color=style['axline_color'],alpha=style['axline_alpha'],
                linestyle=style['axline_linestyle'])
            ax[i,j-1].axhline(0,color=style['axline_color'],alpha=style['axline_alpha'],
                linestyle=style['axline_linestyle'])
            ax[i,j-1].plot(summary_df['avg_weight_'+strategies[j]],
                summary_df['avg_weight_'+strategies[i]],
                'o',alpha=style['data_alpha'],color=style['data_color_all'])
            ax[i,j-1].set_xlabel(pgt.get_clean_string([strategies[j]])[0],
                fontsize=style['label_fontsize_dense'])
            ax[i,j-1].set_ylabel(pgt.get_clean_string([strategies[i]])[0],
                fontsize=style['label_fontsize_dense'])
            ax[i,j-1].xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
            ax[i,j-1].yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])

    plt.tight_layout()
    if savefig:
        directory = pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_"+"weight_avg_scatter.png"
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_weight_avg_scatter_task0(summary_df, version=None,
    savefig=False,group=None,filetype='.png',plot_error=True):
    '''
        Makes a summary plot of the average weights of task0 against omission 
        weights for each session
        Also computes a regression line, and returns the linear model
    '''

    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(3.75,5))  
    strategies = pgt.get_strategy_list(version)
    style = pstyle.get_style()
    plt.plot(summary_df['avg_weight_task0'],summary_df['avg_weight_omissions1'],
        'o',alpha=style['data_alpha'],color=style['data_color_all'])
    ax.set_xlabel('Avg. '+pgt.get_clean_string(['task0'])[0]+' weight',
        fontsize=style['label_fontsize'])
    ax.set_ylabel('Avg. '+pgt.get_clean_string(['omissions1'])[0]+' weight',
        fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.axvline(0,color=style['axline_color'],
        alpha=style['axline_alpha'],
        ls=style['axline_linestyle'])
    ax.axhline(0,color=style['axline_color'],
        alpha=style['axline_alpha'],
        ls=style['axline_linestyle'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Compute Linear Regression
    x = np.array(summary_df['avg_weight_task0'].values).reshape((-1,1))
    y = np.array(summary_df['avg_weight_omissions1'].values)
    model = LinearRegression(fit_intercept=False).fit(x,y)
    sortx = np.sort(summary_df['avg_weight_task0'].values).reshape((-1,1))
    y_pred = model.predict(sortx)
    ax.plot(sortx,y_pred, 
        color=style['regression_color'], 
        linestyle=style['regression_linestyle'])
    score = round(model.score(x,y),2)

    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group) 
        filename=directory+"summary_"+"weight_avg_scatter_task0"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)
    return model


def plot_session_summary_weight_avg_scatter_task_events(summary_df,event,
    version=None,savefig=False,group=None,filetype='.png'):
    '''
        Makes a scatter plot of each weight against the total number of <event>
        <event> needs to be a session-wise metric

        Raises an exception if event is not a session-wise metric
    '''
    
    # Check if we have a discrete session wise event
    if event in ['hits','image_false_alarm','image_correct_reject','miss',
        'trial_false_alarm','trial_correct_reject']:
        df_event = 'num_'+event
    elif event in ['lick_hit_fraction','lick_fraction','trial_hit_fraction',
        'fraction_engaged']:
        df_event = event
    else:
        raise Exception('Bad event type')
    
    # make figure   
    strategies = pgt.get_strategy_list(version) 
    style = pstyle.get_style()
    fig,ax = plt.subplots(nrows=1,ncols=len(strategies),figsize=(14,3))
    num_sessions = len(summary_df)
    for index, strat in enumerate(strategies):
        ax[index].plot(summary_df[df_event], summary_df['avg_weight_'+strat].values,
            'o',alpha=style['data_alpha'],color=style['data_color_'+strat])
        ax[index].set_xlabel(event,fontsize=style['label_fontsize'])
        ax[index].set_ylabel(pgt.get_clean_string([strat])[0],
            fontsize=style['label_fontsize'])
        ax[index].xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
        ax[index].yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
        ax[index].axhline(0,color=style['axline_color'],
            linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])
        ax[index].spines['top'].set_visible(False)
        ax[index].spines['right'].set_visible(False)

    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_"+"weight_avg_scatter_"+event+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)

def plot_session_summary_multiple_trajectory(summary_df,trajectories, version=None,
    savefig=False,group=None,filetype='.png',event_names=''):
    '''
        Makes a summary plot by plotting the average value of trajectory over the session
        trajectory needs to be a image-wise metric, with 4800 values for each session.

        Raises an exception if trajectory is not on the approved list.  
    '''

    # Check if we have an image wise metric
    good_trajectories = ['omissions1','task0','timing1D','omissions','bias',
        'miss', 'reward_rate','is_change','image_false_alarm','image_correct_reject',
        'lick_bout_rate','RT','engaged','hit','lick_hit_fraction_rate',
        'strategy_weight_index_by_image']
    for trajectory in trajectories:
        if trajectory not in good_trajectories:
            raise Exception('Bad summary variable {}'.format(trajectory))


    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,3)) 
    style = pstyle.get_style()
    colors = pstyle.get_project_colors(trajectories)
    for trajectory in trajectories:
        smooth = trajectory in ['hit','miss','RT','image_false_alarm',
            'image_correct_reject']
        mm = 100

        strategies = pgt.get_strategy_list(version)
        if trajectory in strategies:
            plot_trajectory = 'weight_'+trajectory
        else:
            plot_trajectory = trajectory

        # We have only one group of data
        values = np.vstack(summary_df[plot_trajectory].values)
        mean_values = np.nanmean(values, axis=0)
        std_values = np.nanstd(values, axis=0)
        sem_values = std_values/np.sqrt(len(summary_df))
        if smooth:
            mean_values = pgt.moving_mean(mean_values,mm)
            std_values = pgt.moving_mean(std_values,mm)
            sem_values = pgt.moving_mean(sem_values,mm)          
        ax.plot(mean_values,color=colors[trajectory],
            label=pgt.get_clean_string([trajectory])[0])
        ax.fill_between(range(0,len(mean_values)), mean_values-sem_values, 
            mean_values+sem_values,color=style['data_uncertainty_color'],
            alpha=style['data_uncertainty_alpha'])
 
    ax.set_xlim(0,4800)
    ax.axhline(0, color=style['axline_color'],
        linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])
    labels={
        'strategies':'Weight',
        'strategies_visual':'Weight',
        'strategies_timing':'Weight',
        'task_events':'Fraction',
        'metrics':'Rate',
        'responses':'Response Rate'
        }
    ylabel = labels[event_names]
    ax.set_ylabel(ylabel,fontsize=style['label_fontsize']) 
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.set_xlabel('Image #',fontsize=style['label_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()

    # remove extra axis
    plt.tight_layout()
    
    # Save Figure
    if savefig:
        directory= pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_trajectory_comparison_"+event_names+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)

def plot_session_summary_trajectory(summary_df,trajectory, version=None,
    categories=None,savefig=False,group=None,filetype='.png'):
    '''
        Makes a summary plot by plotting the average value of trajectory over the session
        trajectory needs to be a image-wise metric, with 4800 values for each session.

        Raises an exception if trajectory is not on the approved list.  
    '''

    # Check if we have an image wise metric
    good_trajectories = ['omissions1','task0','timing1D','omissions','bias',
        'miss', 'reward_rate','is_change','image_false_alarm','image_correct_reject',
        'lick_bout_rate','RT','engaged','hit','lick_hit_fraction_rate',
        'strategy_weight_index_by_image']
    if trajectory not in good_trajectories:
        raise Exception('Bad summary variable {}'.format(trajectory))

    smooth = trajectory in ['RT','image_false_alarm','image_correct_reject',
        'hit','miss']
    mm = 100

    strategies = pgt.get_strategy_list(version)
    if trajectory in strategies:
        plot_trajectory = 'weight_'+trajectory
    else:
        plot_trajectory = trajectory

    # make figure    
    fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(6,3)) 
    style = pstyle.get_style()

    if categories is None:
        # We have only one group of data
        values = np.vstack(summary_df[plot_trajectory].values)
        mean_values = np.nanmean(values, axis=0)
        std_values = np.nanstd(values, axis=0)
        sem_values = std_values/np.sqrt(len(summary_df))
        if smooth:
            mean_values = pgt.moving_mean(mean_values,mm)
            std_values = pgt.moving_mean(std_values,mm)
            sem_values = pgt.moving_mean(sem_values,mm)          
        ax.plot(mean_values,color=style['data_color_all'])
        ax.fill_between(range(0,len(mean_values)), mean_values-sem_values, 
            mean_values+sem_values,color=style['data_uncertainty_color'],
            alpha=style['data_uncertainty_alpha'])
    else:
        # We have multiple groups of data
        groups = summary_df[categories].unique()
        colors = pstyle.get_project_colors(keys=groups)
        for index, g in enumerate(groups):
            df = summary_df.query(categories +' == @g')
            values = np.vstack(df[plot_trajectory].values)
            mean_values = np.nanmean(values, axis=0)
            std_values = np.nanstd(values, axis=0)
            sem_values = std_values/np.sqrt(len(df))
            if smooth:
                mean_values = pgt.moving_mean(mean_values,mm)
                std_values = pgt.moving_mean(std_values,mm)
                sem_values = pgt.moving_mean(sem_values,mm)          
            ax.plot(mean_values,color=colors[g])
            if type(df.iloc[0][categories]) in [bool, np.bool_]:
                if g:
                    label = pgt.get_clean_string([categories])[0]
                else:
                    label = 'not '+pgt.get_clean_string([categories])[0]
            else:
                    label = pgt.get_clean_string([g])[0]
            ax.fill_between(range(0,len(mean_values)), mean_values-sem_values, 
                mean_values+sem_values,color=colors[g],
                alpha=style['data_uncertainty_alpha'], 
                label=label)
 
    ax.set_xlim(0,4800)
    ax.axhline(0, color=style['axline_color'],
        linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])
    ax.set_ylabel(pgt.get_clean_string([trajectory])[0],
        fontsize=style['label_fontsize']) 
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.set_xlabel('Image #',fontsize=style['label_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if categories is not None:
        plt.legend()
        extra = '_by_'+categories
    else:
        extra =''

    # remove extra axis
    plt.tight_layout()
    
    # Save Figure
    if savefig:
        directory= pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_"+"trajectory_"+trajectory+extra+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_session_summary_roc(summary_df,version=None,savefig=False,group=None,
    filetype=".png"):
    '''
        Make a summary plot of the histogram of AU.ROC values for all sessions 
    '''
    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    ax.set_xlim(0.5,1)
    ax.hist(summary_df['session_roc'],bins=25,
        color=style['data_color_all'], alpha = style['data_alpha'])
    ax.set_ylabel('Count', fontsize=style['label_fontsize'])
    ax.set_xlabel('ROC-AUC', fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    meanscore =summary_df['session_roc'].median()
    ax.plot(meanscore, ax.get_ylim()[1],'rv')
    ax.axvline(meanscore,color=style['regression_color'], 
        linestyle=style['regression_linestyle'],
        alpha=0.75)
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_"+"roc"+filetype
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


def plot_static_comparison(summary_df, version=None,savefig=False,group=None,
    filetype='.png'):
    '''
        Top Level function for comparing static and dynamic logistic regression
         using ROC scores
    
        Computes the values with :
            get_all_static_roc
            get_static_roc
            get_static_design_matrix
        plots with:
            plot_static_comparison_inner
             
    '''
    summary_df = get_all_static_roc(summary_df, version)
    plot_static_comparison_inner(summary_df,version=version, savefig=savefig, 
        group=group,filetype=filetype)
    plot_session_summary_roc_comparison(summary_df, version=version,
        savefig=savefig, group=group,filetype=filetype)
    return summary_df 

def plot_session_summary_roc_comparison(summary_df,version=None,savefig=False,group=None,
    filetype=".png"):
    '''
        Make a summary plot of the histogram of AU.ROC values for all sessions 
    '''
    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    ax.set_xlim(0.5,1)
    bins = np.arange(0.5,1,.02)
    h=ax.hist(summary_df['session_roc'],bins=bins,
        color=style['data_color_all'], alpha = style['data_alpha'],
        label='Dynamic Model')

    ax.set_ylabel('Count', fontsize=style['label_fontsize'])
    ax.set_xlabel('Area under ROC', fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    meanscore =summary_df['session_roc'].median()
    ax.plot(meanscore, ax.get_ylim()[1],'rv')
    ax.axvline(meanscore,color=style['regression_color'], 
        linestyle=style['regression_linestyle'],
        alpha=0.75)
    plt.ylim(top=ax.get_ylim()[1])
    ax.hist(summary_df['static_session_roc'],bins=bins,
        color='k',alpha=style['data_alpha'],label='Static Model')

    plt.legend()
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_roc_comparison"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)



def plot_static_comparison_inner(summary_df,version=None, savefig=False,
    group=None,filetype='.png'): 
    '''
        Plots static and dynamic ROC comparisons

        Called by plot_static_comparison
    
    '''
    fig,ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    plt.plot(summary_df['static_session_roc'],summary_df['session_roc'],'o',
        color=style['data_color_all'],alpha=style['data_alpha'])
    plt.plot([0.5,1],[0.5,1],color=style['axline_color'],
        alpha=style['axline_alpha'], linestyle=style['axline_linestyle'])
    plt.ylabel('Dynamic ROC',fontsize=style['label_fontsize'])
    plt.xlabel('Static ROC',fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_static_comparison"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)


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


def scatter_df(summary_df, key1, key2, categories= None, version=None,
    flip1=False,flip2=False,cindex=None, savefig=False,group=None,
    plot_regression=False,plot_axis_lines=False,filetype='.png',figsize=(6.5,5)):
    '''
        Generates a scatter plot of two session-wise metrics against each other. The
        two metrics are defined by <key1> and <key2>. Additionally, a third metric can
        be used to define the color axis using <cindex>
        
        summary_df (pandas df) 
        key1, (string, must be column of summary_df)
        key2, (string, must be column of summary_df)
        categories, (string) column of summary_df with discrete values to 
            seperately scatter
        version, (behavior model version)
        flip1, (bool) flips the sign of key1
        flip2, (bool) flips the sign of key2       
        cindex, (string, must be column of summary_df)
        savefig, (bool) saves the figure
        group, (string) determines the subdirectory to save the figure, does not perform
            data selection on summary_df
        plot_regression, (bool) plots a regression line and returns the model
        plot_axis_lines, (bool) plots horizontal and vertical axis lines
    '''
    
    assert (categories is None) or (cindex is None), \
        "Cannot have both categories and cindex"

    # Make Figure
    fig,ax = plt.subplots(figsize=figsize)
    style = pstyle.get_style()
    if categories is not None:
        groups = summary_df[categories].unique()
        colors = pstyle.get_project_colors(groups)
        for index, g in enumerate(groups): 
            df = summary_df.query(categories+'==@g')
            vals1 = df[key1].values
            vals2 = df[key2].values            
            if flip1:
                vals1 = -vals1
            if flip2:
                vals2 = -vals2
            plt.plot(vals1,vals2,'o',color=colors[g],alpha=style['data_alpha'],
                label=pgt.get_clean_string([g])[0])  
        plt.legend() 
    else:
        # Get data
        vals1 = summary_df[key1].values
        vals2 = summary_df[key2].values
        if flip1:
            vals1 = -vals1
        if flip2:
            vals2 = -vals2

        if  cindex is None:
           plt.plot(vals1,vals2,'o',color=style['data_color_all'],
                alpha=style['data_alpha'])
        else:
            scat = ax.scatter(vals1,vals2,c=summary_df[cindex],cmap='plasma')
            cbar = fig.colorbar(scat, ax = ax)
            clabel = pgt.get_clean_string([cindex])[0]
            cbar.ax.set_ylabel(clabel,fontsize=style['colorbar_label_fontsize'])
            cbar.ax.tick_params(labelsize=style['colorbar_ticks_fontsize'])
    label_keys = pgt.get_clean_string([key1, key2])
    plt.xlabel(label_keys[0],fontsize=style['label_fontsize'])
    plt.ylabel(label_keys[1],fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Plot a best fit linear regression
    if plot_regression:    
        x = np.array(vals1).reshape((-1,1))
        y = np.array(vals2)
        model = LinearRegression(fit_intercept=True).fit(x,y)
        sortx = np.sort(vals1).reshape((-1,1))
        y_pred = model.predict(sortx)
        plt.plot(sortx,y_pred, color=style['regression_color'], 
            linestyle=style['regression_linestyle'])
        score = round(model.score(x,y),2)
        print('R^2 between '+str(key1)+', '+str(key2)+': '+str(score))
 
    # Plot horizontal and vertical axis lines
    if plot_axis_lines:
        plt.axvline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],
            alpha=style['axline_alpha'])
        plt.axhline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],
            alpha=style['axline_alpha'])

    # Save the figure
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        if categories is not None:
            filename = directory+'scatter_'+key1+'_by_'+key2+'_split_by_'+categories+\
                filetype
        elif cindex is None:
            filename = directory+'scatter_'+key1+'_by_'+key2+filetype
        else:
            filename = directory+'scatter_'+key1+'_by_'+key2+'_with_'+cindex+\
                '_colorbar'+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)

    if plot_regression:
        return model


def plot_df_groupby(summary_df, key, groupby, savefig=False, version=None, 
    group=None,hline=0,filetype='.svg'):
    '''
    Plots the average value of <key> after splitting the data by <groupby>

    summary_df, (pandas dataframe)
    key, (string) must be session-wise column of summary_df
    groupby, (string) must be categorical column of summary_df
    savefig, (bool) saves the figures
    version, (string) model version
    group, (string) saves the figure as a subdirectory does not perform data selection
    '''

    # Data selection
    means = summary_df.groupby(groupby)[key].mean()
    sem = summary_df.groupby(groupby)[key].sem()
    names = np.array(summary_df.groupby(groupby)[key].mean().index) 

    # Make figure
    fig,ax = plt.subplots()
    colors = sns.color_palette("hls",len(means))
    defined_colors = pstyle.get_colors()
    style = pstyle.get_style()
    for index, m in enumerate(means):
        if names[index] in defined_colors:
            c = defined_colors[names[index]]
        else:
            c = colors[index]
        plt.plot([index-0.5,index+0.5], [m, m],'-',color=c,linewidth=4)
        plt.plot([index, index],[m-sem.iloc[index], m+sem.iloc[index]],'-',color=c)
    ax.set_xticks(np.arange(0,len(names)))
    ax.set_xticklabels(pgt.get_clean_string(names),rotation=0,
        fontsize=style['axis_ticks_fontsize'])
    ax.axhline(hline, color=style['axline_color'],
        linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])
    plt.ylabel(pgt.get_clean_string([key])[0],fontsize=style['label_fontsize'])
    plt.xlabel(pgt.get_clean_string([groupby])[0], fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])

    # Do significance testing 
    if len(means) == 2:
        groups = summary_df.groupby(groupby)
        vals = []
        for name, grouped in groups:
            vals.append(grouped[key])
        pval =  ttest_ind(vals[0],vals[1],nan_policy='omit')
        ylim = plt.ylim()[1]
        r = plt.ylim()[1] - plt.ylim()[0]
        sf = .075
        offset = 2 
        plt.plot([0,1],[ylim+r*sf, ylim+r*sf],'-',
            color=style['stats_color'],alpha=style['stats_alpha'])
        plt.plot([0,0],[ylim, ylim+r*sf], '-',
            color=style['stats_color'],alpha=style['stats_alpha'])
        plt.plot([1,1],[ylim, ylim+r*sf], '-',
            color=style['stats_color'],alpha=style['stats_alpha'])
     
        if pval[1] < 0.05:
            plt.plot(.5, ylim+r*sf*1.5,'k*',color=style['stats_color'])
        else:
            plt.text(.5,ylim+r*sf*1.25, 'ns',color=style['stats_color'])

    # Save figure
    if savefig:
        directory = pgt.get_directory(version,subdirectory='figures',group=group)
        filename = directory+'average_'+key+'_groupby_'+groupby+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)

def histogram_df_by_experience(summary_df, stages, key, nbins=12,density=False,
    experience_type='experience_level',version=None, savefig=False, group=None, 
    strict_experience=True,filetype='.svg'):

    if strict_experience:
        print('Limiting to strict experience')
        summary_df = summary_df.query('strict_experience').copy()

    # Set up Figure
    fix, ax = plt.subplots(figsize=(4,4))
    style = pstyle.get_style()
 
    # Get the stage values paired by container
    matched_df = get_df_values_by_experience(summary_df, 
        stages,key,experience_type=experience_type,how='inner')
    matched_df['diff'] = matched_df[stages[1]]-matched_df[stages[0]]

    counts,edges = np.histogram(matched_df['diff'].values,nbins)
    plt.hist(matched_df['diff'], bins=edges,density=density, 
        color=style['data_color_all'], alpha = style['data_alpha'])

    meanscore = matched_df['diff'].mean()
    marker = ax.get_ylim()[1]
    ax.plot(meanscore, marker,'rv',markersize=10)

    # Clean up
    plt.axvline(0,color=style['axline_color'],
        linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])
    plt.ylabel('Count',fontsize=style['label_fontsize'])
    plt.xlabel('$\Delta$ '+pgt.get_clean_string([key])[0]+'\n'+stages[1]+' - '+stages[0],
        fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save Figure
    if savefig:
        extra=''
        if strict_experience:
            extra='strict_'
        directory = pgt.get_directory(version, subdirectory='figures',group=group)
        filename = directory + 'histogram_df_by_'+extra+experience_type+'_'+key+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)



def scatter_df_by_experience(summary_df,stages, key,
    experience_type='experience_level', version=None,savefig=False,group=None,
    strict_experience=True,filetype='.svg'):
    ''' 
        Scatter session level metric <key> for two sessions matched from the same mouse.
        Sessions are matched by <stages> of <experience_type>
    
        experience_type should be 'session_number' or 'experience_level' 
        
    '''
    if strict_experience:
        print('Limiting to strict experience')
        summary_df = summary_df.query('strict_experience').copy()

    # Set up Figure
    fix, ax = plt.subplots(figsize=(4,4))
    style = pstyle.get_style()
 
    # Get the stage values paired by container
    matched_df = get_df_values_by_experience(summary_df, 
        stages,key,experience_type=experience_type)
    plt.plot(matched_df[stages[0]],matched_df[stages[1]],'o',
        color=style['data_color_all'], alpha=style['data_alpha'])

    # Add diagonal axis line
    xlims = plt.xlim()
    ylims = plt.ylim()
    all_lims = np.concatenate([xlims,ylims])
    lims = [np.min(all_lims), np.max(all_lims)]
    plt.plot(lims,lims, 
        color=style['axline_color'],
        linestyle=style['axline_linestyle'],
        alpha=style['axline_alpha'])

    # clean up
    stage_names = pgt.get_clean_session_names(stages)
    plt.xlabel(stage_names[0],fontsize=style['label_fontsize'])
    plt.ylabel(stage_names[1],fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # add significance
    title_key = pgt.get_clean_string([key])[0]
    plt.title(title_key)
    pval = ttest_rel(matched_df[stages[0]],matched_df[stages[1]],nan_policy='omit')
    ylim = plt.ylim()[1]
    if pval[1] < 0.05:
        plt.title(title_key+": *")
    else:
        plt.title(title_key+": ns")
    plt.tight_layout()    

    # Save figure
    if savefig:
        extra =''
        if strict_experience:
            extra = 'strict_'
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename = directory+'scatter_by_'+extra+experience_type+'_'+key+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)


def get_df_values_by_experience(summary_df, stages, key,
    experience_type='experience_level',how='outer'):
    '''
        Filters summary_df for matched sessions, then returns a dataframe with the 
            column <key> for matched sessions. 
        
        summary_df, (dataframe), table of all data
        stages, (list of two experience levels) if there are multiple sessions 
            with the same experience level, it takes the last of the first stage, 
            and the first of the second stage. 
        key, (string, column name in summary_df) the metric to return
        experience_type (string, column name in summary_df) 
            the column to use for stage matching 
        how, (string, must be 'how','inner','left',right). Pandas command to 
            determine how to handle missing values across mice. how='outer' 
            returns incomplete mice with NaNs. 'inner' only returns complete mice
    '''
    x = stages[0]
    y = stages[1]
    s1df = summary_df.query(experience_type+' == @x').\
        drop_duplicates(keep='last',subset='mouse_id').\
        set_index(['mouse_id'])[key]
    s2df = summary_df.query(experience_type+' == @y').\
        drop_duplicates(keep='first',subset='mouse_id').\
        set_index(['mouse_id'])[key]
    s1df.name=x
    s2df.name=y

    full_df = pd.merge(s1df,s2df,on='mouse_id',how=how) 
    return full_df


def histogram_df(summary_df, key, categories = None, version=None, group=None, 
    savefig=False,nbins=20,ignore_nans=False,density=False,filetype='.png'):
    '''
        Plots a histogram of <key> split by unique values of <categories>
        summary_df (dataframe)
        key (string), column of summary_df
        categories (string), column of summary_df with discrete values
        version (int) model version
        group (string), subset of data, does not perform data selection
        savefig (bool), whether to save figure or not
        nbins (int), number of bins for histogram
    '''
    if (ignore_nans) & (np.any(summary_df[key].isnull())):
        print('Dropping rows with NaNs')
        summary_df = summary_df.dropna(subset=[key]).copy()

    # Plot Figure
    fig, ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    counts,edges = np.histogram(summary_df[key].values,nbins)
    if categories is None:
        # We have only one group of data
        plt.hist(summary_df[key].values, bins=edges,density=density, 
            color=style['data_color_all'], alpha = style['data_alpha'])
    else:
        # We have multiple groups of data
        groups = summary_df[categories].unique()
        colors = pstyle.get_project_colors(keys=groups)
        for index, g in enumerate(groups):
            df = summary_df.query(categories +' == @g')
            plt.hist(df[key].values, bins=edges,density=density,
                alpha=style['data_alpha'], color=colors[g],
                label=pgt.get_clean_string([g])[0])

    # Clean up
    plt.axvline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],
        alpha=style['axline_alpha'])
    plt.ylabel('Count',fontsize=style['label_fontsize'])
    plt.xlabel(pgt.get_clean_string([key])[0],fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if 'fraction' in key:
        ax.set_xlim(right=1)

    if categories is not None:
        plt.legend()
    plt.tight_layout()

    # Save Figure
    if savefig:
        if categories is None:
            category_label =''
        else:
            category_label = '_split_by_'+categories 
        directory = pgt.get_directory(version, subdirectory='figures',group=group)
        filename = directory + 'histogram_df_'+key+category_label+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)


def plot_summary_df_by_date(summary_df,key,version=None,savefig=False,
    group=None,tick_labels_by=4):
    '''
        Plots values of <key> sorted by date of aquisition
        tick_labels_by (int) how frequently to plot xtick labels
    '''
    summary_df = summary_df.sort_values(by=['date_of_acquisition'])
    fig, ax = plt.subplots(figsize=(8,4))
    style = pstyle.get_style()
    plt.plot(summary_df.date_of_acquisition,
        summary_df.strategy_dropout_index,'o',
        color=style['data_color_all'],alpha=style['data_alpha'])
    plt.axhline(0, color=style['axline_color'],alpha=style['axline_alpha'], 
        linestyle=style['axline_linestyle'])
    ax.set_xticks(summary_df.date_of_acquisition.values[::tick_labels_by])
    labels = [x[0:10] for x in summary_df.date_of_acquisition.values[::tick_labels_by]]
    ax.set_xticklabels(labels,rotation=90,fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.ylabel('Strategy Dropout Index',fontsize=style['label_fontsize'])
    plt.xlabel('Date of Acquisition',fontsize=style['label_fontsize'])
    plt.tight_layout()

    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename =directory+'df_by_date_'+key+'.png'
        print('Figure saved to: '+filename)
        plt.savefig(filename)


def plot_engagement_analysis(summary_df,version,levels=10, savefig=False,group=None,
    filetype='.svg'):
    ''' 
        Plots a density plot of activity in reward_rate vs lick_bout_rate space
        Then plots histograms of lick_bout_rate and reward_rate
    '''

    # Organize data
    lick_bout_rate = np.hstack(summary_df['lick_bout_rate'].values) 
    lick_bout_rate = lick_bout_rate[~np.isnan(lick_bout_rate)] 
    reward_rate = np.hstack(summary_df['reward_rate'].values) 
    reward_rate = reward_rate[~np.isnan(reward_rate)] 
    threshold = pgt.get_engagement_threshold()

    # Setup figure
    fig,ax = plt.subplots(ncols=2,nrows=2,figsize=(9,4))
    gs = ax[0,0].get_gridspec()
    for a in ax[:,0]:
        a.remove()
    bigax= fig.add_subplot(gs[:,0])
    style = pstyle.get_style()

    # Plot Density plot
    sns.kdeplot(x=lick_bout_rate[0:-1:100], y=reward_rate[0:-1:100],
        levels=levels,ax=bigax)
    bigax.set_ylabel('Reward Rate (Rewards/s)',fontsize=style['label_fontsize'])
    bigax.set_xlabel('Lick Bout Rate (Bouts/s)',fontsize=style['label_fontsize'])
    bigax.set_xlim(0,.5)
    bigax.set_ylim(0,.1)
    bigax.set_aspect(aspect=5)
    bigax.plot([0,.5],[threshold, threshold], color=style['annotation_color'],
        alpha=style['annotation_alpha'],label='Engagement Threshold')
    bigax.legend(loc='upper right')
    bigax.tick_params(axis='both',labelsize=style['axis_ticks_fontsize'])
    bigax.spines['top'].set_visible(False)
    bigax.spines['right'].set_visible(False)

    # Plot histogram of reward rate
    ax[0,1].hist(reward_rate, bins=100,density=True)
    ax[0,1].set_xlim(0,.1)
    ax[0,1].set_ylabel('Density',fontsize=style['label_fontsize'])
    ax[0,1].set_xlabel('Reward Rate',fontsize=style['label_fontsize'])
    
    ax[0,1].spines['top'].set_visible(False)
    ax[0,1].spines['right'].set_visible(False)
    ax[0,1].axvline(threshold,color=style['annotation_color'],
        alpha=style['annotation_alpha'],label='Engagement Threshold (1 Reward/120s)')
    ax[0,1].legend(loc='upper right') 
    ax[0,1].tick_params(axis='both',labelsize=style['axis_ticks_fontsize'])

    # Plot histogram of lick bout rate
    ax[1,1].hist(lick_bout_rate, bins=100,density=True)
    ax[1,1].set_xlim(0,.5)
    ax[1,1].set_ylabel('Density',fontsize=style['label_fontsize'])
    ax[1,1].set_xlabel('Lick Bout Rate',fontsize=style['label_fontsize'])
    ax[1,1].tick_params(axis='both',labelsize=style['axis_ticks_fontsize'])
    ax[1,1].spines['top'].set_visible(False)
    ax[1,1].spines['right'].set_visible(False)
    plt.tight_layout()

    # Save Figure
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename =directory+'engagement_analysis'+filetype
        plt.savefig(filename)
        print('Figure saved to: '+filename)


def plot_engagement_landscape(summary_df,version, savefig=False,group=None,
    bins=100,cmax=1000,filetype='.svg'):
    '''
        Plots a heatmap of the lick-bout-rate against the reward rate
        The threshold for engagement is annotated 
        
        Try these settings:
        bins=100, cmax=1000
        bins=250, cmax=500
        bins=500, cmax=150
    '''

    # Organize data
    lick_bout_rate = np.concatenate(summary_df['lick_bout_rate'].values)
    reward_rate = np.concatenate(summary_df['reward_rate'].values)
    nan_vec = np.isnan(lick_bout_rate) | np.isnan(reward_rate)
    lick_bout_rate = lick_bout_rate[~nan_vec]
    reward_rate = reward_rate[~nan_vec]

    # Make Plot
    fig, ax = plt.subplots(figsize=(5,5))
    h= plt.hist2d(lick_bout_rate, reward_rate, bins=bins,cmax=cmax,cmap='magma')
    style = pstyle.get_style()
    plt.xlabel('Lick Bout Rate (bouts/sec)',fontsize=style['label_fontsize'])
    plt.ylabel('Reward Rate (rewards/sec)',fontsize=style['label_fontsize'])
    ax.tick_params(axis='both',labelsize=style['axis_ticks_fontsize'])
    plt.ylim(top=.10)
    plt.xlim(right=.5)
    plt.tight_layout()
   
    # Add arrows to mark the engagement threshold 
    engagement_threshold = pgt.get_engagement_threshold()
    ax.annotate('',xy=(0,engagement_threshold),xycoords='data',
        xytext=(-.05,engagement_threshold), arrowprops=dict(
        arrowstyle='->',color=style['annotation_color'],
        lw=style['annotation_linewidth']))
    ax.annotate('',xy=(.5,engagement_threshold),xycoords='data',
        xytext=(.55,engagement_threshold), arrowprops=dict(
        arrowstyle='->',color=style['annotation_color'],
        lw=style['annotation_linewidth']))
   
    # Save the figure
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename =directory+'engagement_landscape'+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)


def RT_by_group(summary_df,version,bins=44,ylim=None,
    groups=['visual_strategy_session','not visual_strategy_session'],
    engaged='engaged',labels=['visual','timing'],change_only=False,
    density=True,savefig=False,group=None,filetype='.png'):
    ''' 
        Plots a distribution of response times (RT) in ms for each group in groups. 
        bins, number of bins to use. 44 prevents aliasing
        groups, logical queries to execute on summary_df
        labels, labels for each query
        engaged (bool) look at engaged or disengaged behavior
        change_only (bool) look at all images, or just change images
        density (bool) normalize each to a density rather than raw counts 
    '''

    # Set up figure
    plt.figure(figsize=(6.5,5))
    colors=pstyle.get_project_colors(labels)
    style = pstyle.get_style()
    label_extra=''
    if engaged=='engaged':
        label_extra=' engaged'
    elif engaged=='disengaged':
        label_extra=' disengaged'
    if change_only:
        label_extra+=', change only'

    # Iterate over groups   
    for gindex, g in enumerate(groups):
        RT = []
        for index, row in summary_df.query(g).iterrows():
            vec = row['engaged']
            if engaged=='engaged':
                vec[np.isnan(vec)] = False
                vec = vec.astype(bool)
            elif engaged=='disengaged':
                vec[np.isnan(vec)] = True
                vec = ~vec.astype(bool)
            else:
                vec[:] = True
                vec = vec.astype(bool)
            if change_only:
                c_vec = row['is_change']
                c_vec[np.isnan(c_vec)]=False
                vec = vec & c_vec.astype(bool)
            RT.append(row['RT'][vec]) 

        # Convert to ms from seconds
        RT = np.hstack(RT)*1000

        # Plot distribution of this groups response times
        label = labels[gindex]+label_extra
        plt.hist(RT, color=colors[labels[gindex]],alpha=1/len(groups),
            label=label,bins=bins,density=density,range=(0,750))

    # Clean up plot
    plt.xlim(0,750)
    if ylim is not None:
        plt.ylim(top=ylim)
    plt.axvspan(0,250,facecolor=style['background_color'],
        alpha=style['background_alpha'],edgecolor=None,zorder=1)   
    plt.ylabel('Density',fontsize=style['label_fontsize'])
    plt.xlabel('Response latency from image onset (ms)',
        fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.legend(fontsize=style['axis_ticks_fontsize'])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save figure
    if savefig:
        filename = '_'.join(labels).lower().replace(' ','_')
        if engaged=='engaged':
            filename += '_engaged'
        elif engaged=='disengaged':
            filename += '_disengaged'
        if change_only:
            filename += '_change_images'
        else:
            filename += '_all_images'
        directory = pgt.get_directory(version,subdirectory='figures',group=group)
        filename = directory+'RT_by_group_'+filename+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)


def RT_by_engagement(summary_df,version,bins=44,change_only=False,density=False,
    savefig=False,group=None,filetype='.svg'):
    ''' 
        Plots a distribution of response times (RT) in ms for engaged and 
        disengaged behavior 
        bins, number of bins to use. 44 prevents aliasing
        change_only (bool) look at all images, or just change images
        density (bool) normalize each to a density rather than raw counts 
    '''

    # Aggregate data
    RT_engaged = []
    for index, row in summary_df.iterrows():
        vec = row['engaged']
        vec[np.isnan(vec)] = False
        vec = vec.astype(bool)
        if change_only:
            c_vec = row['is_change']
            c_vec[np.isnan(c_vec)]=False
            vec = vec & c_vec.astype(bool)
        RT_engaged.append(row['RT'][vec])
    RT_disengaged = []
    for index, row in summary_df.iterrows():
        vec = row['engaged']
        vec[np.isnan(vec)] = True
        vec = ~vec.astype(bool)
        if change_only:
            c_vec = row['is_change']
            c_vec[np.isnan(c_vec)]=False
            vec = vec & c_vec.astype(bool)
        RT_disengaged.append(row['RT'][vec])

    # Convert to ms from seconds 
    RT_engaged = np.hstack(RT_engaged)*1000
    RT_disengaged = np.hstack(RT_disengaged)*1000
   
    # Bin data 
    hist_eng, bin_edges_eng = np.histogram(RT_engaged, bins=bins, range=(0,750))     
    hist_dis, bin_edges_dis = np.histogram(RT_disengaged, bins=bins, range=(0,750))
    if density:
        total = len(RT_engaged) + len(RT_disengaged)
        hist_eng = hist_eng/total
        hist_dis = hist_dis/total
    bin_centers_eng = 0.5*np.diff(bin_edges_eng)+bin_edges_eng[0:-1]
    bin_centers_dis = 0.5*np.diff(bin_edges_dis)+bin_edges_dis[0:-1]

    # Set up figure style
    plt.figure(figsize=(6.5,5))
    colors = pstyle.get_project_colors()
    style = pstyle.get_style()
    if change_only:
        label_extra =', change only'
    else:
        label_extra = ''

    # Plot
    plt.bar(bin_centers_eng, hist_eng,color=colors['engaged'],alpha=.5,
        label='Engaged'+label_extra,width=np.diff(bin_edges_eng)[0])
    plt.bar(bin_centers_dis, hist_dis,color=colors['disengaged'],alpha=.5,
        label='Disengaged'+label_extra,width=np.diff(bin_edges_dis)[0])

    # Clean up plot
    if density:
        plt.ylabel('% of all responses',fontsize=style['label_fontsize'])
    else:
        plt.ylabel('count',fontsize=style['label_fontsize'])
    plt.xlim(0,750)
    plt.axvspan(0,250,facecolor=style['background_color'],
        alpha=style['background_alpha'],edgecolor=None,zorder=1)   
    plt.xlabel('Response latency from image onset (ms)',fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.legend(fontsize=style['axis_ticks_fontsize'])
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save
    if savefig:
        directory = pgt.get_directory(version,subdirectory='figures',group=group)
        filename = directory + 'RT_by_engagement'
        if change_only:
            filename += '_change_images'+filetype
        else:
            filename += '_all_images'+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)


def pivot_df_by_experience(summary_df,key='strategy_dropout_index',
    pivot='session_number',mean_subtract=True):
    '''
        pivoted summary_df to look at <key> across different experience levels in <pivot>
        mean_subtract (bool), subtract the average value of <key> across experience 
            level for each mouse
        
        If there are multiple sessions of an experience level for a mouse the values of 
            <key> are averaged together 
        If a mouse does not have an experience level, then the values are NaN
    '''
    x = summary_df[['mouse_id',pivot,key]]
    x_pivot = pd.pivot_table(x,values=key,index='mouse_id',columns=[pivot])

    if mean_subtract:
        experience_levels = x_pivot.columns.values
        x_pivot['mean'] = x_pivot.mean(axis=1)
        for level in experience_levels:
            x_pivot[level] = x_pivot[level] - x_pivot['mean']

    return x_pivot

def plot_pivoted_df_by_experience(summary_df, key,version,flip_index=False,
    experience_type='experience_level', mean_subtract=True,savefig=False,group=None,
    strict_experience=True,full_mouse=True,filetype='.svg'):
    '''
        Plots the average value of <key> across experience levels relative to the average
        value of <key> for each mouse 
    '''
    if strict_experience:
        print('limiting to strict experience')
        summary_df = summary_df.query('strict_experience').copy()

    # Get pivoted data
    if flip_index:
        summary_df = summary_df.copy()
        summary_df[key] = -summary_df[key]
    x_pivot = pivot_df_by_experience(summary_df, key=key,
        mean_subtract=mean_subtract,pivot=experience_type)
    
    if full_mouse:
        print('limiting to full mice')
        x_pivot = x_pivot.dropna()

    # Set up Figure
    fig, ax = plt.subplots(figsize=(4,4))
    levels = np.sort(list(set(x_pivot.columns) - {'mean'}))
    colors = pstyle.get_project_colors(levels)
    style = pstyle.get_style()
    w=.45

    # Plot each stage
    for index,val in enumerate(levels):
        m = x_pivot[val].mean()
        s = x_pivot[val].std()/np.sqrt(len(x_pivot))
        plt.plot([index-w,index+w],[m,m],linewidth=4,color=colors[val])
        plt.plot([index,index],[m+s,m-s],linewidth=1,color=colors[val])
    
    # Add Statistics

    ylim = plt.ylim()[1]
    r = plt.ylim()[1] - plt.ylim()[0]
    if experience_type=='session_number':
        stats = test_significance_by_experience(x_pivot,[3,4],[1,2],ax,ylim,r)

    elif experience_type=='experience_level':
        stats = test_significance_by_experience(x_pivot,['Familiar','Novel 1'],
            [0,1],ax,ylim,r)
        stats = test_significance_by_experience(x_pivot,['Novel 1','Novel >1'],
            [1,2],ax,ylim,r)
        ylim = plt.ylim()[1]
        r = plt.ylim()[1] - plt.ylim()[0]
        stats = test_significance_by_experience(x_pivot,['Familiar','Novel >1'],
            [0,2],ax,ylim,r)

    # Clean up Figure
    label = pgt.get_clean_string([key])[0]
    plt.ylabel('$\Delta$ '+label,fontsize=style['label_fontsize'])
    plt.xlabel('Session #',fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.xticks(range(0,len(levels)),levels,
        fontsize=style['axis_ticks_fontsize'])
    ax.axhline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],
        alpha=style['axline_alpha'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    # Save Figure
    if savefig:
        extra=''
        if strict_experience:
            extra ='strict_'
        directory = pgt.get_directory(version,subdirectory='figures',group=group)  
        filename = directory+'relative_by_'+extra+experience_type+'_'+key+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)

def test_significance_by_experience(x_pivot,g,i,ax,ylim,r):
    style = pstyle.get_style()
    i1 = i[0]
    i2 = i[1]
    mid = (i2-i1)/2+i1
    pval = ttest_ind(x_pivot[g[0]].values, x_pivot[g[1]].values,nan_policy='omit')

    sf = .075
    offset = 2 
    ax.plot([i1,i2],[ylim+r*sf, ylim+r*sf],'-',
        color=style['stats_color'],alpha=style['stats_alpha'])
    ax.plot([i1,i1],[ylim, ylim+r*sf], '-',
        color=style['stats_color'],alpha=style['stats_alpha'])
    ax.plot([i2,i2],[ylim, ylim+r*sf], '-',
        color=style['stats_color'],alpha=style['stats_alpha']) 
    if pval[1] < 0.05:
        ax.plot(mid, ylim+r*sf*1.5,'*',color=style['stats_color'])
    else:
        ax.text(mid,ylim+r*sf*1.25, 'ns',color=style['stats_color'])

    return pval

def plot_segmentation_schematic(session,savefig=False, version=None):
    # Annotate licks and bouts if not already done
    if 'bout_number' not in session.licks:
        pm.annotate_licks(session)
    if 'bout_start' not in session.stimulus_presentations:
        pm.annotate_bouts(session)
    if 'reward_rate' not in session.stimulus_presentations:
        pm.annotate_image_rolling_metrics(session)


    session.stimulus_presentations['images_since_last_lick'] = \
        session.stimulus_presentations.groupby(\
        session.stimulus_presentations['bout_end'].cumsum()).cumcount(ascending=True)
    session.stimulus_presentations['timing_input'] = \
        [x+1 for x in session.stimulus_presentations['images_since_last_lick'].\
        shift(fill_value=0)]

    style = pstyle.get_style()
    xmin = 570.5
    xmax = 592
    fig, ax = plt.subplots(figsize=(8,2.5))

    ax.set_xlim(xmin,xmax)
    yticks = []
    ytick_labels=[]
    tt = .85
    bb = .5
    y1 = .25
    y1_h = .10
    y2 = y1-y1_h-.02
    ax.set_ylim([y2-.02,1])
    xticks = []
    xlabels = []
    for index, row in session.stimulus_presentations.iterrows():
        if (row.start_time > xmin) & (row.start_time < xmax):
            xticks.append(row.start_time+.125)
            if row.in_lick_bout:
                xlabels.append('')       
            else:
                xlabels.append(row.timing_input)
            if not row.omitted:
                # Plot stimulus band
                ax.axvspan(row.start_time,row.stop_time, 
                    alpha=0.1,color='k', label='image')
            else:
                # Plot omission line
                plt.axvline(row.start_time, linestyle='--',linewidth=1.5,
                    color=style['schematic_omission'],label='omission')

            # Plot licked image
            if row.licked:
                r = patches.Rectangle((row.start_time,y1),.75,y1_h,
                    facecolor='gray',alpha=.5)
                ax.add_patch(r)

            if row.in_lick_bout:
                r = patches.Rectangle((row.start_time,y2),.75,y1_h,
                    facecolor='gray',alpha=.5)
                ax.add_patch(r)

            # plot bout_start/end image
            if row.bout_start:
                ax.plot(row.start_time+.1875, y1+y1_h*.5, 'k^',alpha=.5)
            if row.bout_end:
                ax.plot(row.start_time+.5625, y1+y1_h*.5, 'kv',alpha=.5)

    yticks.append(y2+y1_h*.5)
    ytick_labels.append('in bout')
    yticks.append(y1+y1_h*.5)
    ytick_labels.append('image aligned')

    # Label licking
    yticks.append((tt-bb)*.5+bb)
    ytick_labels.append('licks')
    licks_df = session.licks.query('timestamps > @xmin').\
        query('timestamps < @xmax').copy()
    bouts = licks_df.bout_number.unique()
    bout_colors = sns.color_palette('hls',8)
    for b in bouts:
        ax.vlines(licks_df[licks_df.bout_number == b].timestamps,
            bb,tt,alpha=1,linewidth=2,color=bout_colors[np.mod(b,len(bout_colors))])
    # Label bout starts and ends
    ax.plot(licks_df.groupby('bout_number').first().timestamps, 
        (tt+.05)*np.ones(np.shape(licks_df.groupby('bout_number').\
        first().timestamps)), 'kv',alpha=.5,markersize=8)
    yticks.append(tt+.05)
    ytick_labels.append('bout start')
    ax.plot(licks_df.groupby('bout_number').last().timestamps, 
        (bb-.05)*np.ones(np.shape(licks_df.groupby('bout_number')\
        .first().timestamps)), 'k^',alpha=.5,markersize=8)
    yticks.append(bb-.05)
    ytick_labels.append('bout end')

    style = pstyle.get_style() 
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels,fontsize=style['axis_ticks_fontsize'])
    ax.set_xticks(xticks)
    xtick_labels = ['6']+[str(x) for x in np.arange(0,len(xticks)-1)]
    ax.set_xticklabels(xlabels,fontsize=style['axis_ticks_fontsize'])
    ax.set_xlabel('Images since end of last licking bout',
        fontsize=style['label_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

# Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename=directory+"segmentation_example.svg"
        plt.savefig(filename)
        print('Figured saved to: '+filename)
    
   
def plot_session(session,x=None,xStep=5,label_bouts=True,label_rewards=True,
    detailed=False,fit=None,mean_center_strategies=True):
    '''
        Visualizes licking, lick bouts, and rewards compared to stimuli
        press < or > to scroll left or right 

        label_bouts, colors bout segmentations
        label_rewards, annotations reward times
        detailed, annotates auto-rewards, licks that trigger rewards, and
            image-wise annotations of rewards,bout start/stop, hit/miss/FA/CR, in-bout
        fit, adds the strategy inputs
        mean_center_strategies (bool) Shows the mean centered or (0,1) strategies
    '''

    # Annotate licks and bouts if not already done
    if 'bout_number' not in session.licks:
        pm.annotate_licks(session)
    if 'bout_start' not in session.stimulus_presentations:
        pm.annotate_bouts(session)
    if 'reward_rate' not in session.stimulus_presentations:
        pm.annotate_image_rolling_metrics(session)

    # Set up figure
    fig,ax  = plt.subplots()  
    if fit is None:
        fig.set_size_inches(12,4)  
        ax.set_ylim([0, 1])
    else:
        fig.set_size_inches(12,5.6)   
        ax.set_ylim([-.5, 1])
    style = pstyle.get_style()

    # Determine window to plot
    if x is None:
        x = np.floor(session.licks.loc[0].timestamps)-1
        x = [x,x+25]
    elif len(x) ==1:
        x = [x[0],x[0]+25]
    ax.set_xlim(x[0],x[1])
    min_x = x[0]-250
    max_x = x[1]+500

    # Set up y scaling
    tt= .7
    bb = .3
    yticks = []
    ytick_labels = []    

    # Draw all stimulus presentations
    for index, row in session.stimulus_presentations.iterrows():
        if (row.start_time > min_x) & (row.start_time < max_x):
            if not row.omitted:
                # Plot stimulus band
                ax.axvspan(row.start_time,row.stop_time, 
                    alpha=0.1,color='k', label='image')
            else:
                # Plot omission line
                plt.axvline(row.start_time, linestyle='--',linewidth=1.5,
                    color=style['schematic_omission'],label='omission')

            # Plot image change
            if row.is_change:
                ax.axvspan(row.start_time,row.stop_time, alpha=0.5,
                    color=style['schematic_change'], label='change image')
            
            # Plot licked image
            if detailed & row.licked:
                r = patches.Rectangle((row.start_time,.10),.75,.05,
                    facecolor='gray',alpha=.5)
                ax.add_patch(r)

            # Plot rewarded image
            if detailed & row.rewarded:
                r = patches.Rectangle((row.start_time,.15),.75,.05,
                    facecolor='red',alpha=.5)
                ax.add_patch(r)
    
            # plot bout_start/end image
            if detailed & row.bout_start:
                ax.plot(row.start_time+.1875, .125, 'k^',alpha=.5)
            if detailed & row.bout_end:
                ax.plot(row.start_time+.5625, .125, 'kv',alpha=.5)

            # Plot image-wise annotations
            if detailed & (row.change_with_lick==1):
                r = patches.Rectangle((row.start_time,.05),.75,.05,
                    facecolor='red',alpha=.5)
                ax.add_patch(r)
            if detailed & (row.change_without_lick==1):
                r = patches.Rectangle((row.start_time,.05),.75,.05,
                    facecolor='blue',alpha=.5)
                ax.add_patch(r)
            if detailed & (row.non_change_with_lick==1):
                r = patches.Rectangle((row.start_time,.05),.75,.05,
                    facecolor='green',alpha=.5)
                ax.add_patch(r)
            if detailed & (row.non_change_without_lick==1):
                r = patches.Rectangle((row.start_time,.05),.75,.05,
                    facecolor='yellow',alpha=.5)
                ax.add_patch(r)
            if detailed & (row.in_lick_bout):
                r = patches.Rectangle((row.start_time,.00),.75,.05,
                    facecolor='gray',alpha=.5)
                ax.add_patch(r)

    # Add y labels
    if detailed: 
        yticks.append(.125)
        ytick_labels.append('Stimulus licked')
        yticks.append(.175)
        ytick_labels.append('Stimulus rewarded')
        yticks.append(.075)
        ytick_labels.append('Hit/Miss/FA/CR')
        yticks.append(.025)
        ytick_labels.append('In Bout')


    # Label licking
    yticks.append(.5)
    ytick_labels.append('licks')
    if label_bouts:
        # Label the licking bouts as different colors
        bouts = session.licks.bout_number.unique()
        bout_colors = sns.color_palette('hls',8)
        for b in bouts:
            ax.vlines(session.licks[session.licks.bout_number == b].timestamps,
                bb,tt,alpha=1,linewidth=2,color=bout_colors[np.mod(b,len(bout_colors))])
        yticks.append(.5)
        ytick_labels.append('licks')

        # Label bout starts and ends
        ax.plot(session.licks.groupby('bout_number').first().timestamps, 
            (tt+.05)*np.ones(np.shape(session.licks.groupby('bout_number').\
            first().timestamps)), 'kv',alpha=.5,markersize=8)
        yticks.append(tt+.05)
        ytick_labels.append('bout start')
        ax.plot(session.licks.groupby('bout_number').last().timestamps, 
            (bb-.05)*np.ones(np.shape(session.licks.groupby('bout_number')\
            .first().timestamps)), 'k^',alpha=.5,markersize=8)
        yticks.append(bb-.05)
        ytick_labels.append('bout end')
       
        if detailed: 
            # Label the licks that trigger rewards
            ax.plot(session.licks.query('rewarded').timestamps, 
                (tt)*np.ones(np.shape(session.licks.query('rewarded').timestamps)), 
                'rx',alpha=.5,markersize=8)       
            yticks.append(tt)
            ytick_labels.append('reward trigger')

    else:
        # Just label the licks one color
        ax.vlines(session.licks.timestamps,bb,tt,alpha=1,linewidth=2,color ='k')

    # Add Rewards
    if label_rewards:
        ax.plot(session.rewards.timestamps,
            np.zeros(np.shape(session.rewards.timestamps.values))+0.9, 
            'rv', label='reward',markersize=8)
        yticks.append(.9)
        ytick_labels.append('rewards')


        if detailed:
            # Label rewarded bout starts
            ax.plot(session.licks.query('bout_rewarded == True').\
                groupby('bout_number').first().timestamps, 
                (tt+.05)*np.ones(np.shape(session.licks.\
                query('bout_rewarded == True').groupby('bout_number').\
                first().timestamps)), 'rv',alpha=.5,markersize=8)

            # Label auto rewards
            ax.plot(session.rewards.query('autorewarded').timestamps,
                np.zeros(np.shape(session.rewards.query('autorewarded').\
                timestamps.values))+0.95, 
                'rv', label='auto reward',markersize=8,markerfacecolor='w')
            yticks.append(.95)
            ytick_labels.append('auto rewards')
   
    # Add the strategies 
    if fit is not None:
        if mean_center_strategies:
            # Plot mean centered strategies
            task0_s = fit['psydata']['inputs']['task0'][:,0]*.075 -.15
            omissions_s = fit['psydata']['inputs']['omissions'][:,0]*.075 -.25
            omissions1_s = fit['psydata']['inputs']['omissions1'][:,0]*.075 -.35
            timing1D_s = fit['psydata']['inputs']['timing1D'][:,0]*.075 -.45
    
            colors = {True:'darkgray',False:'lightcoral'}
            edge_color={True:'k',False:'red'}
            for index, row in fit['psydata']['df'].reset_index().iterrows():
                if (row.start_time > min_x) & (row.start_time < max_x):
                    h = .15+task0_s[index]
                    r = patches.Rectangle((row.start_time,-.15),.75,h,
                        facecolor=colors[h>0],alpha=.75,edgecolor=edge_color[h>0])       
                    ax.add_patch(r)
                    h = .25+omissions_s[index]
                    r = patches.Rectangle((row.start_time,-.25),.75,h,
                        facecolor=colors[h>0],alpha=.75,edgecolor=edge_color[h>0])
                    ax.add_patch(r)
                    h = .35+omissions1_s[index]
                    r = patches.Rectangle((row.start_time,-.35),.75,h,
                        facecolor=colors[h>0],alpha=.75,edgecolor=edge_color[h>0])
                    ax.add_patch(r)
                    h = .45+timing1D_s[index]
                    r = patches.Rectangle((row.start_time,-.45),.75,h,
                        facecolor=colors[h>0],alpha=.75,edgecolor=edge_color[h>0])
                    ax.add_patch(r)
        else:
            # plot (0,1) strategies
            fit['psydata']['df']['task0_s'] = \
                (fit['psydata']['df']['task0']*.075)-.15
            fit['psydata']['df']['omissions_s']=\
                (fit['psydata']['df']['omissions']*.075)-.25
            fit['psydata']['df']['omissions1_s']=\
                (fit['psydata']['df']['omissions1']*.075)-.35
            fit['psydata']['df']['timing1D_s']=\
                (fit['psydata']['df']['timing1D']*.075)-.375

            for index, row in fit['psydata']['df'].iterrows():
                if (row.start_time > min_x) & (row.start_time < max_x):
                    r = patches.Rectangle((row.start_time,-.15),.75,.15+row.task0_s,
                        facecolor='darkgray',alpha=.75,edgecolor='k')
                    ax.add_patch(r)
                    r = patches.Rectangle((row.start_time,-.25),.75,.25+row.omissions_s,
                        facecolor='darkgray',alpha=.75,edgecolor='k')
                    ax.add_patch(r)
                    r = patches.Rectangle((row.start_time,-.35),.75,.35+row.omissions1_s,
                        facecolor='darkgray',alpha=.75,edgecolor='k')
                    ax.add_patch(r)
                    r = patches.Rectangle((row.start_time,-.45),.75,.45+row.timing1D_s,
                        facecolor='darkgray',alpha=.75,edgecolor='k')
                    ax.add_patch(r)

        # Clean up strategies
        ax.axhline(-.15,color='k',alpha=.2,linestyle='-')
        ax.axhline(-.25,color='k',alpha=.2,linestyle='-')
        ax.axhline(-.35,color='k',alpha=.2,linestyle='-')
        ax.axhline(-.45,color='k',alpha=.2,linestyle='-')
        yticks.append(-.125)
        ytick_labels.append(pgt.get_clean_string(['task0'])[0])
        yticks.append(-.225)
        ytick_labels.append(pgt.get_clean_string(['omissions'])[0])
        yticks.append(-.325)
        ytick_labels.append(pgt.get_clean_string(['omissions1'])[0])
        yticks.append(-.425)
        ytick_labels.append(pgt.get_clean_string(['timing1D'])[0])

    # Clean up plots
    ax.set_xlabel('time (s)',fontsize=style['label_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize']) 
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels,fontsize=style['axis_ticks_fontsize'])
    plt.tight_layout()
   
    # Set up responsive scrolling 
    def on_key_press(event):
        x = ax.get_xlim()
        xmin = x[0]
        xmax = x[1]
        if event.key=='<' or event.key==',' or event.key=='left': 
            xmin -= xStep
            xmax -= xStep
        elif event.key=='>' or event.key=='.' or event.key=='right':
            xmin += xStep
            xmax += xStep
        ax.set_xlim(xmin,xmax)
        plt.draw()
    kpid = fig.canvas.mpl_connect('key_press_event', on_key_press)

    return fig, ax


def plot_session_metrics(session, plot_list = ['reward_rate','lick_hit_fraction',\
    'd_prime','hit_rate'],interactive=True,plot_example=False,version=None):
    '''
        options for plot list:
        plot_list = ['reward_rate','lick_bout_rate','lick_hit_fraction',
        'd_prime','criterion','hit_rate','miss_rate','false_alarm','correct_reject']
    '''

    # Annotate licks and bouts if not already done
    if 'bout_number' not in session.licks:
        pm.annotate_licks(session)
    if 'bout_start' not in session.stimulus_presentations:
        pm.annotate_bouts(session)
    if 'reward_rate' not in session.stimulus_presentations:
        pm.annotate_image_rolling_metrics(session)


    # Set up Figure with two axes
    if plot_example:
        width=12 
    else:
        width=12 
    pre_horz_offset = 1         # Left hand margin
    post_horz_offset = .25      # Right hand margin
    height = 4
    vertical_offset = .6       # Bottom margin
    fixed_height = .75         # height of fixed axis
    gap = .05                   # gap between plots
    top_margin = .25
    variable_offset = fixed_height+vertical_offset+gap 
    variable_height = height-variable_offset-top_margin
    fig = plt.figure(figsize=(width,height))

    # Bottom Axis
    h = [Size.Fixed(pre_horz_offset),Size.Fixed(width-pre_horz_offset-post_horz_offset)]
    v = [Size.Fixed(vertical_offset),Size.Fixed(fixed_height)]
    divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
    fax=fig.add_axes(divider.get_position(),
            axes_locator=divider.new_locator(nx=1,ny=1)) 

    # Top axis
    v = [Size.Fixed(variable_offset),Size.Fixed(variable_height)]
    divider = Divider(fig, (0,0,1,1),h,v,aspect=False)
    ax = fig.add_axes(divider.get_position(), 
            axes_locator=divider.new_locator(nx=1,ny=1),
            sharex=fax) 
   
    # Set up limits and colors
    colors = pstyle.get_project_colors(['d_prime','criterion','false_alarm',
        'hit','miss','correct_reject','lick_hit_fraction'])
    style = pstyle.get_style()

    # Plot licks and rewards on bottom axis 
    for index, row in session.stimulus_presentations.iterrows():
        if row.bout_start:
            fax.axvspan(index,index+1, 0,.333,
                        alpha=0.5,color='k')
        if row.rewarded:
            fax.axvspan(index,index+1, .666,1,
                        alpha=0.5,color='r')
        elif row.is_change:
            fax.axvspan(index,index+1, .333,.666,
                        alpha=0.5,color='b')
    yticks = [.125,.375,.625]
    yticks = [.165,.5,.835]

    ytick_labels = ['Licked','Miss','Hit'] 
    fax.set_yticks(yticks)
    fax.set_yticklabels(ytick_labels,fontsize=style['axis_ticks_fontsize'])
    fax.spines['top'].set_visible(False)
    fax.spines['right'].set_visible(False)

    # Plot Engagement state
    if not plot_example:
        engagement_labels = session.stimulus_presentations['engaged'].values
        engagement_labels=[0 if x else 1 for x in engagement_labels]
        change_point = np.where(~(np.diff(engagement_labels) == 0))[0]
        change_point = np.concatenate([[0], change_point, [len(engagement_labels)]])
        plotted = np.zeros(2,)
        labels = ['engaged','disengaged']
        for i in range(0, len(change_point)-1):
            if plotted[engagement_labels[change_point[i]+1]]:
                ax.axvspan(change_point[i],change_point[i+1],edgecolor=None,
                    facecolor=colors[labels[engagement_labels[change_point[i]+1]]], 
                    alpha=0.2)
            else:
                plotted[engagement_labels[change_point[i]+1]] = True
                ax.axvspan(change_point[i],change_point[i+1],edgecolor=None,
                    facecolor=colors[labels[engagement_labels[change_point[i]+1]]], 
                    alpha=0.2,label=labels[engagement_labels[change_point[i]+1]])
    
        # Add Engagement threshold
        ax.axhline(pgt.get_engagement_threshold(),linestyle=style['axline_linestyle'],
            alpha=style['axline_alpha'], color=style['axline_color'],
            label='Engagement Threshold (1 Reward/120s)')

    if 'reward_rate' in plot_list:
        # Plot Reward Rate
        reward_rate = session.stimulus_presentations.reward_rate
        ax.plot(reward_rate,color=colors['reward_rate'],label='Reward Rate (Rewards/S)')

    if 'prediction' in plot_list:
        prediction = session.stimulus_presentations.prediction
        ax.plot(prediction, color='red',label='Model Prediction')

    if 'target' in plot_list:
        target = session.stimulus_presentations.target
        ax.plot(target, color=style['data_color_all'],alpha=style['data_alpha'],
            label='Data')

    if 'lick_bout_rate' in plot_list:
        # Plot Lick Bout Rate
        lick_bout_rate = session.stimulus_presentations.bout_rate
        ax.plot(lick_bout_rate,color=colors['lick_bout_rate'],
            label='Lick Bout Rate (Bouts/S)')

    if 'lick_hit_fraction' in plot_list:
        # Plot Lick Hit Fraction Rate
        lick_hit_fraction = session.stimulus_presentations.lick_hit_fraction
        ax.plot(lick_hit_fraction,color=colors['lick_hit_fraction'],
            label='Lick Hit Fraction')

    if 'd_prime' in plot_list:
        # Plot d_prime
        d_prime = session.stimulus_presentations.d_prime
        ax.plot(d_prime,color=colors['d_prime'],label='d\'')

    if 'criterion' in plot_list:
        # Plot criterion
        criterion = session.stimulus_presentations.criterion
        ax.plot(criterion,color=colors['criterion'],label='criterion')

    if 'hit_rate' in plot_list:
        # Plot hit_rate
        hit_rate = session.stimulus_presentations.hit_rate
        ax.plot(hit_rate,color=colors['hit'],label='hit %')

    if 'miss_rate' in plot_list:
        # Plot miss_rate
        miss_rate = session.stimulus_presentations.miss_rate
        ax.plot(miss_rate,color=colors['miss'],label='miss %')

    if 'false_alarm' in plot_list:
        # Plot false_alarm_rate
        false_alarm_rate = session.stimulus_presentations.false_alarm_rate
        ax.plot(false_alarm_rate,color=colors['false_alarm'],label='false alarm %')

    if 'correct_reject' in plot_list:
        # Plot correct_reject_rate
        correct_reject_rate = session.stimulus_presentations.correct_reject_rate
        ax.plot(correct_reject_rate,color=colors['correct_reject'],
            label='correct reject %')
    
    # Clean up top axis
    ax.set_xlim(0,4800)
    if interactive:
        ax.set_ylim([0, 1])
    else:
        ax.set_ylim([0, .1])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if plot_example:
        ax.set_ylabel('licking probability',fontsize=style['label_fontsize'])
    else:
        ax.set_ylabel('rate/sec',fontsize=style['label_fontsize'])
    ax.tick_params(axis='both',labelsize=style['axis_ticks_fontsize'],labelbottom=False)
    ax.xaxis.set_tick_params(length=0)
    ax.legend(loc='upper right')
    
    if interactive & (not plot_example):
        ax.set_title('z/x to zoom in/out, </> to scroll left/right, up/down for ylim')

    # Clean up Bottom axis
    fax.set_xlabel('Image #',fontsize=style['label_fontsize'])
    fax.tick_params(axis='both',labelsize=style['axis_ticks_fontsize'])

    if plot_example:
        directory = pgt.get_directory(version, subdirectory ='figures')
        filename = directory +"example_session.svg"
        print('Figure saved to: '+filename)
        plt.savefig(filename)          

    if (not interactive) or (plot_example):
        return fig

    # Set up responsive scrolling 
    def on_key_press(event):
        x = ax.get_xlim()
        xmin = x[0]
        xmax = x[1]
        y= ax.get_ylim()
        ymax = y[1]
        xStep = np.floor((xmax-xmin)/10)
        if event.key=='<' or event.key==',' or event.key=='left': 
            xmin -= xStep
            xmax -= xStep
            ax.set_xlim(xmin,xmax)
        elif event.key=='>' or event.key=='.' or event.key=='right':
            xmin += xStep
            xmax += xStep
            ax.set_xlim(xmin,xmax)
        elif event.key=='z':
            xmin = xmin+xStep/2
            xmax = xmax-xStep/2
            ax.set_xlim(xmin,xmax)
        elif event.key=='x':
            xmin = xmin-xStep/2
            xmax = xmax+xStep/2
            if (xmin < 0) and (xmax > 4800):
                xmin=0
                xmax=4800
            ax.set_xlim(xmin,xmax)
        elif event.key=='down':
            ymax = ymax*.9
            ax.set_ylim(y[0],ymax)
        elif event.key=='up':
            ymax = ymax*1.1
            ax.set_ylim(y[0],ymax)
        elif event.key=='r':
            ax.set_xlim(0,4800)
            ax.set_ylim(0,1)
        plt.draw()
    kpid = fig.canvas.mpl_connect('key_press_event', on_key_press)

def plot_session_engagement(session,version, savefig=False):
    '''
        Plots the lick_bout_rate, reward_rate, and engagement state for a single session 
    '''
    
    fig = plot_session_metrics(session,interactive=not savefig,
        plot_list=['reward_rate','lick_bout_rate','hit_rate'])

    if savefig:
        directory = pgt.get_directory(version, subdirectory ='session_figures')
        filename = directory +str(behavior_session_id)+'_engagement.png'
        print('Figure saved to: '+filename)
        plt.savefig(filename)   


def plot_image_pair_repetitions(change_df, version,savefig=False, group=None,
    filetype='.svg'):
    ''' 
        Plots a histogram of how often a change between a unique pair of 
        images is repeated in a single session 
    '''
    # get unique pair repeats per session
    counts = change_df.groupby(['behavior_session_id','post_change_image',\
        'pre_change_image']).size().values

    # make figure    
    fig,ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    ax.hist(counts,bins=0.5+np.array(range(0,9)),density=True,rwidth=.9,
        color=style['data_color_all'], alpha = style['data_alpha'])
    ax.set_ylabel('% of image changes', fontsize=style['label_fontsize'])
    ax.set_xlabel('repetitions of each image pair\n per session', 
        fontsize=style['label_fontsize'])
    ax.xaxis.set_ticks(np.array(range(1,9)))
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save figure
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_image_pair_repetitions"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_image_repeats(change_df,version,categories=None,savefig=False, group=None,
    filetype='.png'):
    '''
        Plot the number of image repetitions between image changes. Omissions 
        are counted as an image repetition. 
        
        categories (str) a categorical column in change_df to split the data by
        
    '''
    # Set up Figure 
    fig,ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    bins = 0.5+np.array(range(0,50))
    key = 'image_repeats'

    # Plot data
    if categories is None:
        values = change_df[key]
        ax.hist(values,bins=bins,density=True,color=style['data_color_all'], 
            alpha = style['data_alpha'])
    else:
         groups = change_df[categories].unique()
         colors = pstyle.get_project_colors(keys=groups)
         for index, g in enumerate(groups):
             df = change_df.query(categories +' == @g')
             plt.hist(df[key].values, bins=bins,alpha=style['data_alpha'],
                 color=colors[g],label=pgt.get_clean_string([g])[0],density=True)

    # Clean up Figure
    ax.set_ylabel('% of image changes', fontsize=style['label_fontsize'])
    ax.set_xlabel('repeats between changes', fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if categories is not None:
        plt.legend()
    plt.tight_layout()

    # Save Figure
    if savefig:
        if categories is None:
            category_label =''
        else:
            category_label = '_split_by_'+categories 
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_"+key+category_label+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)

def plot_interlick_interval(licks_df,key='pre_ili',categories = None, version=None, 
    group=None, savefig=False,nbins=40,xmax=20,filetype='.png'):
    '''
        Plots a histogram of <key> split by unique values of <categories>
        licks_df (dataframe)
        key (string), column of licks_df
        categories (string), column of licks_df with discrete values
        version (int) model version
        group (string), subset of data, does not perform data selection
        savefig (bool), whether to save figure or not
        nbins (int), number of bins for histogram
    '''
    
    # Remove NaNs (from start of session) and limit to range defined by xmax
    licks_df = licks_df.dropna(subset=[key]).query('{} < @xmax'.format(key)).copy()

    if categories is not None:
        licks_df = licks_df.dropna(subset=[categories]).copy()
        density=False
    if key =='pre_ili':
        xlabel= 'interlick interval (s)'
        yscale=4
    elif key =='pre_ibi':
        xlabel= 'interbout interval (s)'
        yscale=1.5
    else:
        xlabel=key
        yscale=1.5

    # Plot Figure
    fig, ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    counts,edges = np.histogram(licks_df[key].values,nbins)
    if categories is None:
        # We have only one group of data
        plt.hist(licks_df[key].values, bins=edges, 
            color=style['data_color_all'], alpha = style['data_alpha'])
    else:
        # We have multiple groups of data
        groups = licks_df[categories].unique()
        colors = pstyle.get_project_colors(keys=groups)
        for index, g in enumerate(groups):
            df = licks_df.query(categories +' == @g')
            if (type(g) == bool) or (type(g) == np.bool_):
                if g:
                    label = categories
                else:
                    label = 'not '+categories
            else:
                label = pgt.get_clean_string([g])[0]
            plt.hist(df[key].values, bins=edges,
                alpha=style['data_alpha'], color=colors[g],
                label=label)

    # Clean up
    plt.ylim(top = np.sort(counts)[-2]*yscale)

    plt.xlim(0,xmax)
    plt.axvline(.700,color=style['axline_color'],
        linestyle=style['axline_linestyle'],alpha=style['axline_alpha'],
        label='Licking bout threshold')
    plt.ylabel('Count',fontsize=style['label_fontsize'])
    plt.xlabel(xlabel,fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if categories is not None:
        plt.legend()
    plt.tight_layout()

    # Save Figure
    if savefig:
        if categories is None:
            category_label =''
        else:
            category_label = '_split_by_'+categories 
        directory = pgt.get_directory(version, subdirectory='figures',group=group)
        filename = directory + 'histogram_df_'+key+category_label+filetype
        print('Figure saved to: '+filename)
        plt.savefig(filename)

def plot_chronometric(bouts_df,version,savefig=False, group=None,xmax=8,
    nbins=40,method='chronometric',key='pre_ibi'):
    ''' 
        Plots the % of licking bouts that were rewarded as a function of time since
        last licking bout ended        
    '''
    # Filter data
    bouts_df = bouts_df.dropna(subset=[key]).query('{} < @xmax'.format(key)).copy()
    
    # Compute chronometric
    if method =='chronometric':
        counts, edges = np.histogram(bouts_df[key].values,nbins)
        counts_m, edges_m = np.histogram(\
            bouts_df.query('not bout_rewarded')[key].values, bins=edges)
        counts_h, edges_h = np.histogram(\
            bouts_df.query('bout_rewarded')[key].values, bins=edges)
        centers = edges[0:-1]+np.diff(edges)
        chronometric = counts_h/counts  
        err = 1.96*np.sqrt(chronometric/(1-chronometric)/counts)
        label='Hit %'
    elif method=='hazard':
        print('Warning, this method is very sensitive to xmax')
        counts, edges = np.histogram(bouts_df[key].values,nbins) 
        counts_h, edges_h= np.histogram(\
            bouts_df.query('bout_rewarded')[key].values,bins=edges)
        centers = np.diff(edges) + edges[0:-1]

        pdf = counts/np.sum(counts)
        survivor = 1 - np.cumsum(pdf)
        dex = np.where(survivor > 0.005)[0]
        hazard = pdf[dex]/survivor[dex]
        pdf_hits = counts_h/np.sum(counts)
        hazard_hits = pdf_hits[dex]/survivor[dex]
        centers = centers[dex]
        chronometric=hazard
        err = 1.96*np.sqrt(chronometric/(1-chronometric)/counts[dex])
        label='Hazard Function'

    # Make figure
    fig, ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style() 
    plt.plot(centers, chronometric,color=style['data_color_all'])
    ax.fill_between(centers, chronometric-err, chronometric+err,
        color=style['data_uncertainty_color'],alpha=style['data_uncertainty_alpha'])

    # Clean up
    plt.axvline(.700,color=style['axline_color'],
        linestyle=style['axline_linestyle'],alpha=style['axline_alpha'])
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.ylabel(label,fontsize=style['label_fontsize'])
    plt.xlabel('interbout interval (s)',fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.tight_layout()

    # Save Figure
    if savefig:
        directory = pgt.get_directory(version, subdirectory='figures',group=group)
        if len(bouts_df['behavior_session_id'].unique()) == 1:
            extra = '_'+str(bouts_df.loc[0]['behavior_session_id'])
        if method =='chronometric':
            filename = directory + 'chronometric'+extra+'.png'
        elif method =='hazard':
            filename = directory + 'hazard'+extra+'.png'
        print('Figure saved to: '+filename)
        plt.savefig(filename)


def plot_bout_durations(bouts_df,version, savefig=False, group=None,filetype='.png'):
    '''
        Generates two plots of licking bout durations split by hit or miss
        The first plot is in units of number of licks, the second is in 
        licking bout duration 
    '''
    # Plot duration by number of licks
    fig, ax = plt.subplots(figsize=(5,4))
    style = pstyle.get_style()
    colors = pstyle.get_project_colors(keys=['not rewarded','rewarded'])
    edges = np.array(range(0,np.max(bouts_df['bout_length']+1)))+0.5
    h = plt.hist(bouts_df.query('not bout_rewarded')['bout_length'],
        bins=edges,color=colors['not rewarded'],label='Miss',
        alpha=style['data_alpha'],density=True)
    plt.hist(bouts_df.query('bout_rewarded')['bout_length'],bins=edges,
        color=colors['rewarded'],label='Hit',alpha=style['data_alpha'],
        density=True)
    plt.xlabel('# licks in bout',fontsize=style['label_fontsize'])
    plt.ylabel('Density',fontsize=style['label_fontsize'])
    plt.legend()
    ax.set_xticks(np.arange(0,np.max(bouts_df['bout_length']),5))
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlim(0,50)

    # Save figure
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"bout_duration_licks"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)


    # Plot duration by time
    fig, ax = plt.subplots(figsize=(5,4))
    edges = np.arange(0,5,.1)
    h = plt.hist(bouts_df.query('not bout_rewarded')['bout_duration'],
        bins=edges,color=colors['not rewarded'],label='Miss',
        alpha=style['data_alpha'],density=True)
    plt.hist(bouts_df.query('bout_rewarded')['bout_duration'],bins=h[1],
        color=colors['rewarded'],label='Hit',alpha=style['data_alpha'],
        density=True)
    plt.xlabel('bout duration (s)',fontsize=style['label_fontsize'])
    plt.ylabel('Density',fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend()
    plt.xlim(0,5)

    # Save figure
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"bout_duration_seconds"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)

def compare_across_versions(merged_df, column,versions):
    
    # Set up figure
    fig,ax = plt.subplots(1,2,figsize=(8,4))
    style = pstyle.get_style()

    # extra data
    col1 = merged_df[column+'_'+str(versions[0])]
    col2 = merged_df[column+'_'+str(versions[1])]

    # plot scatter plot
    ax[0].plot(col1,col2,'o',
        alpha=style['data_alpha'],
        color=style['data_color_all'])

    # plot identity line
    min_val = np.min([np.min(col1),np.min(col2)])
    max_val = np.max([np.max(col1),np.max(col2)])
    ax[0].plot([min_val,max_val],[min_val,max_val],
        color=style['axline_color'],
        alpha=style['axline_alpha'],
        linestyle=style['axline_linestyle'])

    # clean up plot
    ax[0].set_xlabel('version '+str(versions[0]),fontsize=style['label_fontsize'])
    ax[0].set_ylabel('version '+str(versions[1]),fontsize=style['label_fontsize'])
    ax[0].xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax[0].yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax[0].set_aspect('equal','box')

    # Plot histogram
    diff = col2-col1
    ax[1].hist(diff, bins=40,color=style['data_color_all']) 
    ax[1].axvline(0,
        color=style['axline_color'],
        linestyle=style['axline_linestyle'],
        alpha=style['axline_alpha'])
    
    # clean up plot
    ax[1].set_xlabel('v'+str(versions[1]) + ' - v'+str(versions[0]),
        fontsize=style['label_fontsize'])
    ax[1].set_ylabel('Count',fontsize=style['label_fontsize'])
    ax[1].xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax[1].yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
   
    # Clean up entire plot 
    plt.suptitle(pgt.get_clean_string([column])[0],fontsize=style['label_fontsize'])
    plt.tight_layout()

def plot_timing_curve(version):
    
    # Compute curve
    x_values = np.arange(1,11)
    format_options = ps.get_format_options(version,{})
    params = format_options['timing_params']
    curve = [ps.timing_sigmoid(x-1,params) for x in x_values]
    
    # Clean up plot
    plt.figure()
    style = pstyle.get_style()
    plt.plot(x_values,curve,'-',color=style['data_color_all'])

    # Plot slope tangent line
    slope = -params[0]/(4*(params[1]))
    plt.plot([3,5],[-.5-slope,-.5+slope],'-',color=style['regression_color'],
        alpha=.75,linestyle=style['regression_linestyle'])

    # Clean up
    plt.ylabel('Weight',fontsize=style['label_fontsize'])
    plt.xlabel('Images since last lick',fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.ylim(-1,0)
    plt.xlim(1,10)
    plt.title('Slope Factor = {}, Midpoint = {} \n Slope at midpoint = {}'.\
        format(params[0],params[1]-1,slope))


def scatter_df_by_mouse(summary_df,key,ckey=None,version=None,savefig=False,group=None,
    filetype='.png'):

    # Make Figure
    fig,ax = plt.subplots(figsize=(6.5,5))
    style = pstyle.get_style()

    # Compute average values for each mouse
    if ckey==key:
        ckey=None
    if ckey is None:
        cols = ['mouse_id',key,'mouse_avg_'+key]
        ckey=key
    else:
        cols = ['mouse_id',key,ckey,'mouse_avg_'+key]

    summary_df['mouse_avg_'+key] = \
        summary_df.groupby('mouse_id')[key].transform('mean')
    df = summary_df[cols].copy()
    df = df.sort_values(by='mouse_avg_'+key)
    mice = df['mouse_id'].unique()
    min_val = df[ckey].min()
    max_val = df[ckey].max()

    # Compute fraction of variance explained by mouse
    total_var = df[key].var()
    df['relative'] =df[key] - df['mouse_avg_'+key]
    within_var = df['relative'].var()
    VE = np.round((1-within_var/total_var)*100,1)
    plt.title('Explained Variance: {}%'.format(VE),fontsize=style['label_fontsize'])
 
    # Iterate across mice
    for index, mouse in enumerate(mice):
        # Filter for this mouse
        mouse_df = df.query('mouse_id ==@mouse')
        mouse_avg = mouse_df.iloc[0]['mouse_avg_'+key]
        
        # Plot average for this mouse
        plt.plot([index-.5,index+.5],[mouse_avg,mouse_avg],'k-')
    
        # Plot each session for this mouse
        plt.scatter(index*np.ones(len(mouse_df)),mouse_df[key],c=mouse_df[ckey],
            cmap='plasma',vmin=min_val,vmax=max_val,s=15) 
    
        # plot background shading
        if np.mod(index,2) == 0:
            ax.axvspan(index-.5,index+.5,color=style['background_color'],
                alpha=style['background_alpha'])
 
    # Clean Up
    ax.set_xlim(-.5,len(mice)-.5)
    ax.axhline(0,color=style['axline_color'],alpha=style['axline_alpha'],
        linestyle=style['axline_linestyle'])
    ax.set_ylabel(pgt.get_clean_string([key])[0],fontsize=style['label_fontsize'])
    ax.set_xlabel('Mice (sorted by average value)', fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.set_xticks(np.arange(0,len(mice)))
    ax.set_xticklabels('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Save figure
    plt.tight_layout()
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"scatter_df_by_mouse_"+key+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)

def plot_engagement_comparison(summary_df,version, savefig=False,group=None,
    bins=45,cmax=.075,rate='RT',xlim=[0,.75],normalize=True):
    '''
        Plots a heatmap of the lick-bout-rate against the reward rate
        The threshold for engagement is annotated 
        
        Try these settings:
        bins=100, cmax=1000
        bins=250, cmax=500
        bins=500, cmax=150
    '''

    # Organize data
    lick_bout_rate = np.concatenate(summary_df[rate].values)
    reward_rate = np.concatenate(summary_df['reward_rate'].values)
    nan_vec = np.isnan(lick_bout_rate) | np.isnan(reward_rate)
    lick_bout_rate = lick_bout_rate[~nan_vec]
    reward_rate = reward_rate[~nan_vec]

    # Make Plot
    fig, ax = plt.subplots(figsize=(5,5))
    h= plt.hist2d(lick_bout_rate, reward_rate, bins=bins,cmap='magma')
    if normalize:
        row_sum = np.sum(h[0],0)
        norm_h = np.vstack([h[0][:,x]/row_sum[x] if row_sum[x] >0 \
            else h[0][:,x]/np.nan for x in range(0,len(row_sum))])
        plt.clf()
        plt.imshow(norm_h,aspect='auto',cmap='magma',origin='lower',
            interpolation='None',extent=[h[1][0],h[1][-1],h[2][0],h[2][-1]],
            vmax=cmax)
        ax = plt.gca()   

    style = pstyle.get_style()
    plt.xlabel('Response Time (sec)',fontsize=style['label_fontsize'])
    plt.ylabel('Reward Rate (rewards/sec)',fontsize=style['label_fontsize'])
    ax.tick_params(axis='both',labelsize=style['axis_ticks_fontsize'])
    plt.title('Row normalized',fontsize=style['label_fontsize'])
    plt.ylim(top=.10)
    xlim = [h[1][0],h[1][-1]]
    plt.xlim(xlim)
    plt.tight_layout()


    # Add arrows to mark the engagement threshold 
    engagement_threshold = pgt.get_engagement_threshold()
    ax.annotate('',xy=(xlim[0],engagement_threshold),xycoords='data',
        xytext=(xlim[0]-.05,engagement_threshold), arrowprops=dict(
        arrowstyle='->',color=style['annotation_color'],
        lw=style['annotation_linewidth']))
    ax.annotate('',xy=(xlim[1],engagement_threshold),xycoords='data',
        xytext=(xlim[1]+.05,engagement_threshold), arrowprops=dict(
        arrowstyle='->',color=style['annotation_color'],
        lw=style['annotation_linewidth']))
   
    # Save the figure
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename =directory+'engagement_comparison.png'
        print('Figure saved to: '+filename)
        plt.savefig(filename)


def plot_strategy_examples(session, version=None, savefig=False,max_events=20):
    if 'bout_number' not in session.licks:
        pm.annotate_licks(session)
    if 'bout_start' not in session.stimulus_presentations:
        pm.annotate_bouts(session)
    if 'reward_rate' not in session.stimulus_presentations:
        pm.annotate_image_rolling_metrics(session)

    fig, ax = plt.subplots(2,2,figsize=(6,4))

    plot_strategy_examples_inner(ax[0,0],session, max_events, 'task')
    plot_strategy_examples_inner(ax[0,1],session, max_events, 'timing')
    plot_strategy_examples_inner(ax[1,0],session, max_events, 'omission')
    plot_strategy_examples_inner(ax[1,1],session, max_events, 'post_omission')
    plt.tight_layout()

    # Save the figure
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename =directory+'strategy_examples.svg'
        print('Figure saved to: '+filename)
        plt.savefig(filename)



def plot_strategy_examples_inner(ax,session, max_events, example):
    style = pstyle.get_style()

    # Draw aligning events special to this example
    window = [-2,2]
    if example == 'task':
        ax.axvspan(0,0.25, alpha=0.5, color=style['schematic_change'])
    elif example == 'omission':
        ax.axvline(0, linestyle='--',linewidth=1.5,
            color=style['schematic_omission'])
    elif example == 'post_omission':
        ax.axvline(0, linestyle='--',linewidth=1.5,
            color=style['schematic_omission'])
        window = np.array(window) + .75 
    elif example =='timing':
        ax.axvspan(0,0.25,alpha=0.1,color='k')
        window = [-4,2]
        timing_count = 5
        xticks = np.arange(0,window[0],-.75)[0:timing_count]
        labels = [str(timing_count - x[0]) for x in enumerate(xticks)]
        images = set(np.concatenate([np.arange(-.75,window[0],-.75),\
                np.arange(.75,window[1],.75)]))
        images = list(images - set(xticks))
        xticks = list(xticks)
        for x in images:
            xticks.append(x)
            labels.append('')
        ax.set_xticks(xticks)
        ax.set_xticklabels(labels)

    # Draw all stimulus presentations
    images = np.concatenate([np.arange(-.75,window[0],-.75),\
                np.arange(.75,window[1],.75)])
    for image in images:
        ax.axvspan(image, image+0.25,alpha=0.1,color='k')

    # Get epochs for this example
    session.stimulus_presentations['images_since_last_lick'] = \
        session.stimulus_presentations.groupby(\
        session.stimulus_presentations['bout_end'].cumsum()).cumcount(ascending=True)
    session.stimulus_presentations['timing_input'] = \
        [x+1 for x in session.stimulus_presentations['images_since_last_lick'].\
        shift(fill_value=0)]
    if example == 'task': 
        events = session.stimulus_presentations\
            .query('is_change & bout_start')['start_time'].values
    elif example == 'omission':
        events = session.stimulus_presentations\
            .query('omitted & bout_start & (timing_input>2)')['start_time'].values 
    elif example == 'post_omission':
        session.stimulus_presentations['post_omission'] =\
            session.stimulus_presentations['omitted'].shift(1,fill_value=False)
        events = session.stimulus_presentations\
            .query('post_omission & bout_start &(timing_input>3)')\
            ['start_time'].values - .75 
    elif example == 'timing':
        events = session.stimulus_presentations\
            .query('(timing_input == @timing_count)&bout_start')['start_time'].values

    events = events[0:max_events]   
 
    # Plot licks around each epoch
    for index, e in enumerate(events):       
        plot_lick_raster(ax,index, session, e, window)

    # Clean up labels
    if example == 'task':
        ax.set_xlabel('Time from change (s)',fontsize=style['label_fontsize'])
    elif example == 'omission':
        ax.set_xlabel('Time from omission (s)',fontsize=style['label_fontsize'])
    elif example == 'post_omission':
        ax.set_xlabel('Time from omission (s)',fontsize=style['label_fontsize'])
    elif example == 'timing':
        ax.set_xlabel('Images since end of last\nlicking bout',
        fontsize=style['label_fontsize'])
    ax.set_ylabel('epochs',fontsize=style['label_fontsize'])
    ax.set_yticks([])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def plot_lick_raster(ax, y, session, time, window):
    min_time = time + window[0]
    max_time = time + window[1]
    licks = session.licks.query('(timestamps > @min_time)&(timestamps < @max_time)')
    ax.plot(licks['timestamps']-time,y*np.ones(len(licks)),'k|')
    ax.set_xlim(window)
