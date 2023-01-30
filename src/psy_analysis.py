import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import psy_style as pstyle
import psy_visualization as pv
import psy_general_tools as pgt


## Principal Component Analysis
#######################################################################
def plot_pca_vectors(pca,strategies, version, on, savefig=False, 
    group=None,filetype='.svg'):
    '''
    
    '''
    # Make Figure
    fig,ax = plt.subplots(figsize=(5.5,2))
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
    ax.set_xlabel('strategies',fontsize=style['label_fontsize'])
    ax.set_ylabel('PC weights',fontsize=style['label_fontsize'])
    ax.set_xticks(np.arange(1,ndims+1))
    ax.set_xticklabels(pgt.get_clean_string(strategies))
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+'pca_on_'+on+'_vectors'+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)


def plot_pca_explained_variance(pca,version, on, savefig=False, group=None,
    filetype='.svg',strategy_index_ve = None):
    '''
    
    '''
    # Make Figure
    fig,ax = plt.subplots(figsize=(2.5,2))
    style = pstyle.get_style()

    # Plot
    ndims = len(pca.explained_variance_ratio_)
    plt.plot(np.arange(1,ndims+1),pca.explained_variance_ratio_,'o-',
        color=style['data_color_all'], alpha=style['data_alpha'])
    if strategy_index_ve is not None:
        plt.plot(0,strategy_index_ve,'o',alpha=style['data_alpha'],
        color='k')

    # Clean up
    ax.set_ylabel('explained \nvariance',fontsize=style['label_fontsize'])
    if strategy_index_ve is None:
        ax.set_xticks(np.arange(1,ndims+1))
        ax.set_xlabel('PC #',fontsize=style['label_fontsize'])
    else:
        ax.set_xticks(np.arange(0,ndims+1))
        labels = ['SI']+[str(x) for x in np.arange(1,ndims+1)]
        ax.set_xticklabels(labels)
        ax.set_xlabel('PC #',fontsize=style['label_fontsize'])
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0,1)
    if on != 'dropout':
        ax.set_title('PCA on '+on,fontsize=style['label_fontsize'])
    plt.tight_layout()

    # Save and return
    if savefig:
        directory=pgt.get_directory(version,subdirectory='figures',group=group)
        filename=directory+'pca_on_'+on+'_explained_variance'+filetype
        plt.savefig(filename)
        print('Figured saved to: '+filename)
    

def compute_PCA(summary_df, version, on='dropout',savefig=False, group=None,
    remove_bias=True,filetype='.svg'):


    # Get matrix of data
    strategies = pgt.get_strategy_list(version)
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
    if on != 'dropout':
        extra = '_'+on
    else:
        extra = ''
    for pc in range(0,ndims):
        summary_df['PC'+str(pc+1)+extra] = X[:,pc]

    # Analysis plots
    strategy_index_VE=compute_variance_on_strategy_index(vals, strategies)
    task_timing_VE = np.sum(np.var(vals[:,2:],axis=0))/np.sum(np.var(vals,axis=0))
    print('% Variance along strategy index: {}'.format(strategy_index_VE*100))
    print('% Variance along task/timing: {}'.format(task_timing_VE*100))
    print('% Variance along PC1: {}'.format(pca.explained_variance_ratio_[0]*100))
    print('% Variance along PC2: {}'.format(pca.explained_variance_ratio_[1]*100))
    print('% Variance along PC3: {}'.format(pca.explained_variance_ratio_[2]*100))
    print('% Variance along PC4: {}'.format(pca.explained_variance_ratio_[3]*100))
    print('% variance along PC1+2: {}'.format(np.sum(pca.explained_variance_ratio_[0:2])))
    print('\n')
    plot_pca_explained_variance(pca, version, on, savefig=savefig, 
        group=group, filetype=filetype,strategy_index_ve = strategy_index_VE)
    plot_pca_vectors(pca,strategies, version, on, savefig=savefig, 
        group=group, filetype=filetype)

    # scatter pc1 vs pc2
    pv.scatter_df(summary_df,'PC1'+extra,'PC2'+extra,cindex='strategy_dropout_index',
        version=version, savefig=savefig, group=group,figsize=(4,3.5))

    # scatter pc1 vs strategy index
    pv.scatter_df(summary_df,'PC1'+extra,'strategy_dropout_index',version=version,
        savefig=savefig,group=group,filetype=filetype,figsize=(4,3.5))

    # dropout index by avg mouse
    pv.scatter_df_by_mouse(summary_df,'strategy_dropout_index',version=version,
        savefig=savefig,group=group,filetype=filetype)

def compute_variance_on_strategy_index(vals,strategies):
    t = [1 if x=='task0' else -1 if x=='timing1D' else 0 for x in strategies]
    p = np.array(t)[:,np.newaxis]
    p = p/np.linalg.norm(p)
    x = np.dot(vals,p)
    VE = np.var(x)/np.sum(np.var(vals,axis=0))
    return VE

def compare_PCA(summary_df, version, savefig=False, group=None):
    '''
        Top level function. Performs PCA on dropout scores and avg. weights
        Compares the results
    '''

    compute_PCA(summary_df, version, on='dropout', savefig=savefig, group=group)
    compute_PCA(summary_df, version, on='weights', savefig=savefig, group=group)
    pv.scatter_df(summary_df,'PC1_dropout','PC1_weights',version=version,
        savefig=savefig, group=group)



