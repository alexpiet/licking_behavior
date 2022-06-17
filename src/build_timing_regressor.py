import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import psy_tools as ps
import psy_style as pstyle
import psy_visualization as pv
import psy_general_tools as pgt


def build_timing_regressor():
    '''
        Loads fits with 1-hot timing regressors and finds the best 1D curve
    '''
    df,strategies = get_all_weights()
    ax = plot_summary_weights(df,strategies)
    popt, pcov = compute_average_fit(df,strategies)
    df = fit_each(df, strategies)
    plot_fits(df)
    plot_fits_against_timing_index(df)
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
    fig,ax = plt.subplots(figsize=(4,6))
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
    plt.ylabel('Avg. Weights across each session',fontsize=style['label_fontsize'])
    ax.axhline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],
        alpha=style['axline_alpha'])
    ax.set_xticklabels(pgt.get_clean_string(strategies),
        fontsize=style['axis_ticks_fontsize'], rotation = 90)
    ax.xaxis.tick_top()
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.xlim(-0.5,len(strategies) - 0.5)
    plt.tight_layout()

    
    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+"summary_"+"weights"+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)
    
    return ax

def sigmoid(x,a,b,c,d):
    '''
        four parameter sigmoid function 
    '''
    y = d+(a-d)/(1+(x/c)**b)
    return y

def compute_average_fit(df,strategies):
    '''
        Compute the sigmoid fit to the average weights across sessions 
    '''
    # do fit
    x = np.arange(1,11)
    y = df.mean(axis=0)[strategies]
    popt,pcov = curve_fit(sigmoid, x,y,p0=[0,1,1,-3.5])

    # plot data, fit, and adjusted parameters
    plt.figure()
    style = pstyle.get_style()
    plt.plot(x,y,'o',color='k',alpha=style['data_alpha'],label='average weight')
    plt.plot(x,sigmoid(x,popt[0],popt[1],popt[2],popt[3]),
        color=style['regression_color'],label='best fit')
    plt.plot(x,sigmoid(x,0,-5,4,popt[3]-popt[0]),'b',label='normalized')
    plt.gca().axhline(0,color='k',linestyle='--')
    plt.xlabel('Images since last lick',fontsize=style['label_fontsize'])
    plt.ylabel('Regressor Amplitude',fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.xlim(x[0]-.25,x[-1]+.25)
    plt.legend()
    plt.tight_layout()

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

def plot_fits(df):
    '''
        Scatter plot individual fits
    '''

    plt.figure()
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
    plt.tight_layout()
   
def plot_fits_against_timing_index(df):
    '''
        Scatter plot individual fits against timing dropout
    ''' 
    plt.figure()
    style = pstyle.get_style()
    plt.plot(df['dropout'],df['p3'],'ko')
    plt.ylabel('Intercept')
    plt.xlabel('Timing Dropout')
    plt.gca().axhline(4,color='r',alpha=.25,linestyle='--')
    plt.ylim(bottom=0)
    plt.ylabel('Midpoint',fontsize=style['label_fontsize'])
    plt.xlabel('Timing dropout',fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize']) 
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.tight_layout()   



