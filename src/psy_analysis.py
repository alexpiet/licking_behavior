import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import psy_style as pstyle
import psy_visualization as pv
import psy_general_tools as pgt


## Principal Component Analysis
#######################################################################
def plot_pca_vectors(pca,strategies, version, on, savefig=False, group=None):
    '''
    
    '''
    # Make Figure
    fig,ax = plt.subplots(figsize=(6,2))
    style = pstyle.get_style()

    # Plot
    ndims = len(pca.explained_variance_ratio_)
    xvals = np.arange(1,ndims+1)
    plt.plot(xvals, pca.components_[0,:],'ro-',label='PC #1')
    plt.plot(xvals, pca.components_[1,:],'bo-',label='PC #2')
    ax.axhline(0,linestyle=style['axline_linestyle'],color=style['axline_color'],
        alpha=style['axline_alpha'])
    plt.legend()

    # Clean up
    ax.set_xlabel('Strategies',fontsize=style['label_fontsize'])
    ax.set_ylabel('PC Weights',fontsize=style['label_fontsize'])
    ax.set_xticks(np.arange(1,ndims+1))
    ax.set_xticklabels(pgt.get_clean_string(strategies))
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    plt.tight_layout()

    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+'pca_on_'+on+'_vectors'+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_pca_explained_variance(pca,version, on, savefig=False, group=None):
    '''
    
    '''
    # Make Figure
    fig,ax = plt.subplots()
    style = pstyle.get_style()

    # Plot
    ndims = len(pca.explained_variance_ratio_)
    plt.plot(np.arange(1,ndims+1),pca.explained_variance_ratio_,'o-',
        color=style['data_color_all'], alpha=style['data_alpha'])

    # Clean up
    ax.set_ylabel('Explained Variance',fontsize=style['label_fontsize'])
    ax.set_xlabel('PC #',fontsize=style['label_fontsize'])
    ax.set_xticks(np.arange(1,ndims+1))
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.set_title('PCA on '+on,fontsize=style['label_fontsize'])
    plt.tight_layout()

    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+'pca_on_'+on+'_explained_variance'+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)
    

def compute_PCA(summary_df, version, on='dropout',savefig=False, group=None,
    remove_bias=True):


    # Get matrix of data
    strategies = pgt.get_strategy_list(21)
    if remove_bias:
        strategies.remove('bias')
    if on == 'dropout':
        cols = ['dropout_'+x for x in strategies]
    elif on == 'cv_dropout':
        cols = ['dropout_cv_'+x for x in strategies]
    elif on =='weights':
        cols = ['avg_weight_'+x for x in strategies] 
    vals = summary_df[cols].to_numpy()

    # Do PCA
    pca = PCA()
    pca.fit(vals)
    X = pca.transform(vals)

    # Add to summary_df
    ndims = len(cols)
    for pc in range(0,ndims):
        summary_df['PC'+str(pc+1)+'_'+on] = X[:,pc]

    # Analysis plots
    plot_pca_explained_variance(pca, version, on, savefig=savefig, group=group)
    plot_pca_vectors(pca,strategies, version, on, savefig=savefig, group=group)

    # scatter pc1 vs pc2
    pv.scatter_df(summary_df,'PC1_'+on,'PC2_'+on,cindex='strategy_dropout_index',
        version=version, savefig=savefig, group=group)

    # scatter pc1 vs strategy index
    pv.scatter_df(summary_df,'PC1_'+on,'strategy_dropout_index',version=version,
        savefig=savefig,group=group)

    # dropout index by avg mouse
    pv.scatter_df_by_mouse(summary_df,'strategy_dropout_index',version,savefig=savefig,
        group=group)


def compare_PCA(summary_df, version, savefig=False, group=None):
    '''
        Top level function. Performs PCA on dropout scores and avg. weights
        Compares the results
    '''

    compute_PCA(summary_df, version, on='dropout', savefig=savefig, group=group)
    compute_PCA(summary_df, version, on='weights', savefig=savefig, group=group)
    pv.scatter_df(summary_df,'PC1_dropout','PC1_weights',version=version,
        savefig=savefig, group=group)


## Event Triggered Analysis
#######################################################################
def triggered_analysis(summary_df, version=None,triggers=['hit'],dur=80,
    responses=['weight_timing1D','weight_task0']):
    
    options={
        'subtract_avg':False,
        'subtract_shuffle':True,
        'min_events':50
        }

    eta = compute_triggered_analysis(summary_df, options, triggers, responses, dur)
    plot_triggered_analysis(eta, options, version)
    eta['options'] = options
    return eta

def compute_triggered_analysis(summary_df, options, triggers, responses, dur):
    eta = {}
    for trigger in triggers:
        eta[trigger]={}
        for response in responses:
            stas =[]
            shuffles =[]
            avg_trajectory = np.nanmean(np.vstack(summary_df[response].values),0)
            for index, row in tqdm(summary_df.iterrows()):
                try:
                    mean, shuffle = session_triggered_analysis(row, options, trigger, response,\
                        dur,avg_trajectory)
                    stas.append(mean)
                    shuffles.append(shuffle)
                except Exception as e:
                    pass 
            mean = np.nanmean(stas,0)
            n=np.shape(stas)[0]
            sem = np.nanstd(stas,0)/np.sqrt(n)
            shuffle = np.nanmean(shuffles,0)
            xvalues = (np.arange(0,len(mean))+1)*.75
            eta[trigger][response]={}
            eta[trigger][response]['mean']=mean
            eta[trigger][response]['sem']=sem
            eta[trigger][response]['xvalues']=xvalues
            eta[trigger][response]['shuffle'] = shuffle
    return eta


def session_triggered_analysis(df_row,options, trigger,response, dur,avg_trajectory):
    if np.sum(df_row[trigger] == 1) < options['min_events']:
        raise Exception('not enough events')
    indexes = np.where(df_row[trigger] ==1)[0]
    vals = []
    if options['subtract_avg']:
        residual = df_row[response] - avg_trajectory
    else:
        residual = df_row[response]
    for index in indexes:
        vals.append(get_aligned(residual,index, length=dur))
    if len(vals) >1:
        mean= np.mean(np.vstack(vals),0)
        mean = mean - mean[0]
    else:
        mean = np.array([np.nan]*dur)

    all_shuffles = []
    for n in range(0,100):
        shuffle_trigger = df_row[trigger].copy()
        np.random.shuffle(shuffle_trigger)
        indexes = np.where(shuffle_trigger ==1)[0]
        shuffle_vals = []
        if options['subtract_avg']:
            residual = df_row[response] - avg_trajectory
        else:
            residual = df_row[response]
        for index in indexes:
            shuffle_vals.append(get_aligned(residual,index, length=dur))
        if len(shuffle_vals) >1:
            shuffle_mean= np.mean(np.vstack(shuffle_vals),0)
            shuffle_mean = shuffle_mean - shuffle_mean[0]
        else:
            shuffle_mean = np.array([np.nan]*dur)
        all_shuffles.append(shuffle_mean)
    shuffle_mean = np.nanmean(np.vstack(all_shuffles),0)

    return mean, shuffle_mean


def get_aligned(vector, start, length=4800):

    if len(vector) >= start+length:
        aligned= vector[start:start+length]
    else:
        aligned = np.concatenate([vector[start:], [np.nan]*(start+length-len(vector))])
    return aligned


def plot_triggered_analysis(eta,options, version, savefig=False, group=None):
    
    # Set up figure
    fig,ax = plt.subplots()
    style = pstyle.get_style()

    # Plot
    for trigger in eta.keys():
        for response in eta[trigger].keys():
            mean = eta[trigger][response]['mean']
            sem =  eta[trigger][response]['sem']
            shuffle =  eta[trigger][response]['shuffle']
            xvalues = eta[trigger][response]['xvalues']
            if options['subtract_shuffle']:
                mean = mean - shuffle           
 
            clean_response = pgt.get_clean_string([response])[0]
            plt.plot(xvalues,mean, label=clean_response+' by '+trigger)
            ax.fill_between(xvalues, mean-sem,mean+sem,
                color=style['data_uncertainty_color'],
                alpha=style['data_uncertainty_alpha'])
            plt.ylabel('$\Delta$ Response',fontsize=style['label_fontsize'])    

    # Clean up
    ax.axhline(0,color=style['axline_color'],alpha=style['axline_alpha'],
        linestyle=style['axline_linestyle'])
    plt.xlabel('Time (s)',fontsize=style['label_fontsize'])
    plt.xticks(fontsize=style['axis_ticks_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.legend()
    plt.tight_layout()






