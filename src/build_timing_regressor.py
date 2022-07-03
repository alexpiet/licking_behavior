import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import psy_tools as ps
import psy_style as pstyle
import psy_visualization as pv
import psy_general_tools as pgt
import psy_metrics_tools as pm


def build_timing_schematic(session=None, version=None, savefig=False):
    '''
    
    '''
    if session is None:
        bsid = 951520319
        session = pgt.get_data(bsid)

    # Annotate licks and bouts if not already done
    if 'bout_number' not in session.licks:
        pm.annotate_licks(session)
    if 'bout_start' not in session.stimulus_presentations:
        pm.annotate_bouts(session)
    if 'reward_rate' not in session.stimulus_presentations:
        pm.annotate_image_rolling_metrics(session)
    xmin = 344.25 
    xmax = 350.25
    xmin = 348.75 
    xmax = 355.5
    fig, ax = plt.subplots(figsize=(4,2))
    ax.set_ylim([0, 1]) 
    ax.set_xlim(xmin,xmax)
    yticks = []
    ytick_labels = []   
    tt= .7
    bb = .3
    
    xticks = []
    for index, row in session.stimulus_presentations.iterrows():
        if (row.start_time > xmin) & (row.start_time < xmax):
            xticks.append(row.start_time+.125)
            if not row.omitted:
                # Plot stimulus band
                ax.axvspan(row.start_time,row.stop_time, 
                    alpha=0.1,color='k', label='image')
            else:
                # Plot omission line
                plt.axvline(row.start_time, linestyle='--',linewidth=1.5,
                    color=style['schematic_omission'],label='omission')


    # Label licking
    # Label the licking bouts as different colors
    licks_df = session.licks.query('timestamps > @xmin').\
        query('timestamps < @xmax').copy()
    bouts = licks_df.bout_number.unique()
    bout_colors = sns.color_palette('hls',8)
    for b in bouts:
        ax.vlines(licks_df[licks_df.bout_number == b].timestamps,
            bb,tt,alpha=1,linewidth=2,color=bout_colors[np.mod(b,len(bout_colors))])
    yticks.append(.5)
    ytick_labels.append('licks')

    # Label bout starts and ends
    ax.plot(licks_df.groupby('bout_number').first().timestamps, 
        (tt+.15)*np.ones(np.shape(licks_df.groupby('bout_number').\
        first().timestamps)), 'kv',alpha=.5,markersize=8)
    yticks.append(tt+.15)
    ytick_labels.append('licking \nbout start')
    ax.plot(licks_df.groupby('bout_number').last().timestamps, 
        (bb-.15)*np.ones(np.shape(licks_df.groupby('bout_number')\
        .first().timestamps)), 'k^',alpha=.5,markersize=8)
    yticks.append(bb-.15)
    ytick_labels.append('licking \nbout end')

    style = pstyle.get_style() 
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels,fontsize=style['axis_ticks_fontsize'])
    ax.set_xticks(xticks)
    xtick_labels = ['6']+[str(x) for x in np.arange(0,len(xticks)-1)]
    ax.set_xticklabels(xtick_labels,fontsize=style['axis_ticks_fontsize'])
    ax.set_xlabel('Images since end of last \nlicking bout',
        fontsize=style['label_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename=directory+"Timing_example.svg"
        plt.savefig(filename)
        print('Figured saved to: '+filename)
    

def build_timing_regressor(version=None, savefig=False):
    '''
        Loads fits with 1-hot timing regressors and finds the best 1D curve
    '''
    df,strategies = get_all_weights()
    ax = plot_summary_weights(df,strategies,version=version, savefig=savefig)
    popt, pcov = compute_average_fit(df,strategies,version=version, savefig=savefig)
    df = fit_each(df, strategies)
    plot_fits(df,version=version,savefig=savefig)
    plot_fits_against_timing_index(df,version=version,savefig=savefig)
    return df


def get_all_weights():
    '''
        Load all the average timing weights across sessions
    '''
    # get list of pkl files in directory
    dir="/home/alex.piet/codebase/behavior/psy_fits_v4/"
    files = os.listdir(dir)
    files = [f for f in files if os.path.isfile(dir+f)&('pkl' in f)&\
            ('mouse' not in f)&('clusters' not in f)&('dropouts' not in f)&\
            ('roc' not in f)]
   
    # Make a dataframe to store results 
    strategies = ['timing'+str(x) for x in list(range(1,11))]
    df = pd.DataFrame(columns=strategies)

    # load each session
    for f in tqdm(files):
        filename = dir+f
        fit = ps.load(filename)
        bsid = int(f[:-4])
        weights = ps.get_weights_list(fit['weights'])
        
        # get average value for each regressor
        for index, s in enumerate(weights):
            if s in strategies:
                df.at[bsid,s] = np.mean(fit['wMode'][index,:])

        # get timing dropout
        df.at[bsid, 'dropout'] = get_timing_dropout(fit)

    return df,strategies


def get_timing_dropout(fit):
    '''
        Compute the timing dropout for this fit
    '''
   
    timing_dex = np.where(np.array(fit['labels']) == 'All timing')[0][0]
    full_dex = np.where(np.array(fit['labels']) == 'Full-Task0')[0][0]
    dropout = (1-fit['models'][timing_dex][1]/fit['models'][full_dex][1])*100
    return dropout   


def plot_summary_weights(df,strategies, version=None, savefig=False,
    group=None,filetype='.svg'):
    '''
        Makes a summary plot showing the average weight value for each session
    '''

    # make figure    
    fig,ax = plt.subplots(figsize=(4,4))
    num_sessions = len(df)
    style = pstyle.get_style()
    color = pstyle.get_project_colors(strategies)
    for index, strat in enumerate(strategies):
        ax.plot([index]*num_sessions, df[strat].values,'o',alpha=style['data_alpha'],
            color=color[strat])
        strat_mean = df[strat].mean()
        ax.plot([index-.25,index+.25], [strat_mean, strat_mean], 'k-',lw=3)
        if np.mod(index,2) == 0:
            plt.axvspan(index-.5,index+.5,color=style['background_color'],
                alpha=style['background_alpha'])

    # Clean up
    ax.set_xticks(np.arange(0,len(strategies)))
    plt.ylabel('Avg. Weight',fontsize=style['label_fontsize'])
    ax.axhline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],
        alpha=style['axline_alpha'])
    ax.set_xticklabels(pgt.get_clean_string(strategies),
        fontsize=style['axis_ticks_fontsize'])
    #ax.xaxis.tick_top()
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.xlabel('Images since end of last \nlicking bout',fontsize=style['label_fontsize'])
    plt.xlim(-0.5,len(strategies) - 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim(-4,3)
    plt.tight_layout()

    
    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"Timing_"+"weights"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)
    
    return ax

def sigmoid(x,a,b,c,d):
    '''
        four parameter sigmoid function 
    '''
    y = d+(a-d)/(1+(x/c)**b)
    return y

def compute_average_fit(df,strategies,savefig=False,version=None,
    group=None,filetype='.svg'):
    '''
        Compute the sigmoid fit to the average weights across sessions 
    '''
    # do fit
    x = np.arange(1,11)
    y = df.mean(axis=0)[strategies]
    popt,pcov = curve_fit(sigmoid, x,y,p0=[0,1,1,-3.5])
    print(popt)
    # plot data, fit, and adjusted parameters
    fig,ax = plt.subplots(figsize=(4,4))
    style = pstyle.get_style()
    plt.plot(x,y,'o',color='k',alpha=style['data_alpha'],label='average weight')
    plt.plot(x,sigmoid(x,popt[0],popt[1],popt[2],popt[3]),
        color=style['regression_color'],label='best fit')
    plt.plot(x,sigmoid(x,0,-5,4,popt[3]-popt[0]),'b',label='normalized')
    plt.gca().axhline(0,color='k',linestyle='--')
    plt.xlabel('Images since end of last \nlicking bout',fontsize=style['label_fontsize'])
    plt.ylabel('Regressor Amplitude',fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.xlim(x[0]-.25,x[-1]+.25)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.legend()
    plt.tight_layout()

    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"Timing_regressor"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)

    return popt, pcov
  
def fit_each(df,strategies):
    '''
        Fit each session individually
    '''

    # Iterate over sessions
    for index, row in df.iterrows():
        try:
            # do fit
            x = np.arange(1,11)
            y = row[strategies].values
            p,c = curve_fit(sigmoid, x,y,p0=[0,1,1,-3.5])
                
            # Store results
            df.at[row.name,'p1'] = p[0] 
            df.at[row.name,'p2'] = p[1] 
            df.at[row.name,'p3'] = p[2] 
            df.at[row.name,'p4'] = p[3] 
        except:
            # get errors if the fit doesn't converge
            pass

    return df

def plot_fits(df,version=None, savefig=False, filetype='.svg'):
    '''
        Scatter plot individual fits
    '''

    fig,ax = plt.subplots(figsize=(4,4))
    style = pstyle.get_style()
    plt.plot(np.abs(df['p2']),df['p3'],'ko')
    plt.gca().axvline(5,color='r',alpha=.25,linestyle='--')
    plt.gca().axhline(4,color='r',alpha=.25,linestyle='--')
    plt.xlim(0,20)
    plt.ylim(bottom=0)
    plt.ylabel('Midpoint',fontsize=style['label_fontsize'])
    plt.xlabel('Slope parameter',fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize']) 
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
  
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename=directory+"Timing_parameters"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)
 
def plot_fits_against_timing_index(df,version=None, savefig=False, filetype='.svg'):
    '''
        Scatter plot individual fits against timing dropout
    ''' 
    fig,ax = plt.subplots(figsize=(4,4))
    style = pstyle.get_style()
    plt.plot(df['dropout'],df['p3'],'ko')
    plt.gca().axhline(4,color='r',alpha=.25,linestyle='--')
    plt.ylim(bottom=0)
    plt.ylabel('Midpoint',fontsize=style['label_fontsize'])
    plt.xlabel('Timing dropout',fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize']) 
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()  
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename=directory+"Timing_scatter_midpoint"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)

    fig,ax = plt.subplots(figsize=(4,4))
    style = pstyle.get_style()
    plt.plot(df['dropout'],np.abs(df['p2']),'ko')
    plt.gca().axhline(5,color='r',alpha=.25,linestyle='--')
    plt.ylim(0,20)
    plt.ylabel('Slope parameter',fontsize=style['label_fontsize'])
    plt.xlabel('Timing dropout',fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize']) 
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()  
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures')
        filename=directory+"Timing_scatter_slope"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)




