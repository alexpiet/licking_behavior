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



