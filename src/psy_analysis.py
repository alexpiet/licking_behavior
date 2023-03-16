import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import psy_style as pstyle
import psy_visualization as pv
import psy_general_tools as pgt
import psy_metrics_tools as pm

## Counting/Timing interval analysis
#######################################################################

def build_interval_lick_df(bsid,version):
    session = pgt.get_data(bsid)
    pm.get_metrics(session)

    df = session.stimulus_presentations
    df['first_lick_time'] = [row.licks[0] if len(row.licks) > 0 else np.nan
        for index, row in df.iterrows()]
    df['last_lick_time'] = [row.licks[-1] if len(row.licks) > 0 else np.nan
        for index, row in df.iterrows()]

    print('saving')
    df.to_csv(pgt.get_directory(version, \
        subdirectory='strategy_df')+str(bsid)+'.csv') 



def count_inter_lick_duration(session_df):
    
    # Annotate between bout images 
    session_df['bout_number'].fillna(method='ffill',inplace=True)
    session_df['between_bouts'] = (~session_df['in_lick_bout']) & \
        (~session_df['lick_bout_start'])
    session_df.at[~session_df['between_bouts'],'bout_number'] = np.nan

    # annotate by rewarded bouts
    session_df['post_reward'] = session_df['rewarded']
    session_df.at[~session_df['lick_bout_start'],'post_reward'] = np.nan
    session_df['post_reward'].fillna(method='ffill',inplace=True)
    session_df.at[~session_df['between_bouts'],'post_reward'] = np.nan

    session_df['pre_reward'] = session_df['rewarded']
    session_df.at[~session_df['lick_bout_start'],'pre_reward'] = np.nan
    session_df['pre_reward'].fillna(method='bfill',inplace=True)
    session_df.at[~session_df['between_bouts'],'pre_reward'] = np.nan
     
    # Get images since last lick 
    session_df['images_since_last_lick'] = session_df.groupby(\
        session_df['lick_bout_end'].cumsum()).cumcount(ascending=True)
    session_df.at[session_df['in_lick_bout'],'images_since_last_lick'] = 0    

    # Get images since last reward
    session_df['images_since_last_reward'] = session_df.groupby(\
        session_df['rewarded'].cumsum()).cumcount(ascending=True)
    session_df['post_reward2'] = session_df['images_since_last_reward']<=10
    session_df.at[~session_df['lick_bout_start'],'post_reward2'] = np.nan
    session_df['post_reward2'].fillna(method='ffill',inplace=True)
    session_df.at[~session_df['between_bouts'],'post_reward2'] = np.nan

    # Group by bouts and annotate number of images and whether there was an omission
    bout_df = pd.DataFrame()
    g = session_df.groupby(['bout_number'])
    bout_df['omitted_between_bouts'] = g['omitted'].any()
    bout_df['change_between_bouts'] = g['is_change'].any()
    bout_df['pre_reward'] = g['pre_reward'].any()
    bout_df['post_reward'] = g['post_reward'].any()
    bout_df['images_between_bouts'] = g.size()
 
    # filter omission_df
    #session_df = session_df.join(bout_df.reset_index()\
    #    [['bout_number','change_between_bouts']], on='bout_number')
    #session_df = session_df.\
    #    query('(pre_reward==0)&(post_reward==0)&(not change_between_bouts)') 
 
    return bout_df, session_df

def get_inter_lick_duration(summary_df,version):
    
    dfs = []
    o_dfs = []
    crash = 0
    for index, row in tqdm(summary_df.iterrows(),total=summary_df.shape[0]):
        try:
            strategy_dir = pgt.get_directory(version, subdirectory='strategy_df')
            session_df = pd.read_csv(strategy_dir+str(row.behavior_session_id)+'.csv')
        except Exception as e:
            crash += 1
            print(e)
        else:
            bout_df,s_df = count_inter_lick_duration(session_df)
            bout_df['behavior_session_id'] = row.behavior_session_id 
            dfs.append(bout_df)
            o_dfs.append(s_df)
    print(crash)
    bout_df =  pd.concat(dfs)
    bout_df = bout_df.merge(summary_df[['behavior_session_id','visual_strategy_session']],      on='behavior_session_id')
    omission_df = pd.concat(o_dfs)
    return bout_df, omission_df

def plot_omission_prob(omission_df):
    omission_df.query('(pre_reward ==0)&(post_reward==0)')#&(not change_between_bouts)')
    plt.figure(figsize=(5,4))
    plt.plot(omission_df.groupby('images_since_last_lick')['omitted'].mean().head(14))
    plt.ylabel('Omission prob.',fontsize=16)
    plt.xlabel('Images since end of last licking bout',fontsize=16)
    plt.ylim(bottom=0)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.tight_layout()

def plot_fraction(bout_df):
    z = 1.96
    plt.figure()
    g = bout_df.query('not visual_strategy_session').groupby('images_between_bouts')
    vis = pd.DataFrame()
    vis['p'] = g['omitted_between_bouts'].mean()
    vis['n'] = g.size()
    vis['ci'] = z*np.sqrt(vis['p']*(1-vis['p'])/vis['n'])
    plt.plot(vis['p'],color='blue')
    plt.gca().fill_between(vis.index.values,
        vis['p']-vis['ci'],vis['p']+vis['ci'],color='lightblue')


    g = bout_df.query('visual_strategy_session').groupby('images_between_bouts')
    tim = pd.DataFrame()
    
    tim['p'] = g['omitted_between_bouts'].mean()
    tim['n'] = g.size()
    tim['ci'] = z*np.sqrt(tim['p']*(1-tim['p'])/tim['n'])
    plt.plot(tim['p'],color='blue')
    plt.gca().fill_between(tim.index.values,
        tim['p']-tim['ci'],tim['p']+tim['ci'],color='navajowhite')
    plt.plot(tim['p'],color='darkorange')
    plt.xlim(1,10)
    plt.ylim(0,.3)
    plt.xlabel('# images between licking bouts',fontsize=16)
    plt.ylabel('fraction of intervals with an omission',fontsize=16) 


def plot_inter_lick_duration(bout_df,max_interval = 15,no_rewards=True,cumsum=False):

    if no_rewards:
        bout_df = bout_df.query('(not pre_reward) & (not post_reward) & (not change_between_bouts)')

    fig, ax = plt.subplots(1,2,figsize=(9,4))
 
    omitted = bout_df.query('omitted_between_bouts')
    non_omitted = bout_df.query('not omitted_between_bouts')
    o = pd.DataFrame()
    n = pd.DataFrame()
    o['n'] = omitted.groupby('images_between_bouts').size()
    n['n'] = non_omitted.groupby('images_between_bouts').size()
    if cumsum:
        o['n'] = o['n'].cumsum()
        n['n'] = n['n'].cumsum()
        o['p'] = o['n']/o.iloc[-1]['n']
        n['p'] = n['n']/n.iloc[-1]['n']  
    else:
        o['p'] = o['n']/o['n'].sum()
        n['p'] = n['n']/n['n'].sum()  
    z=1.96 
    o['ci'] = z*np.sqrt(o['p']*(1-o['p'])/o['n'])
    n['ci'] = z*np.sqrt(n['p']*(1-n['p'])/n['n'])
    # Counts
    ax[0].plot(o['n'], color='cyan',label='Omission in interval')  
    ax[0].plot(n['n'], color='gray',label='No omission in interval') 
    ax[0].set_xlabel('# images between licking bouts',fontsize=16)
    if cumsum:
        ax[0].set_ylabel('cumulative counts',fontsize=16)        
    else:
        ax[0].set_ylabel('counts',fontsize=16) 
    ax[0].set_xlim(1,max_interval)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].xaxis.set_tick_params(labelsize=12)
    ax[0].yaxis.set_tick_params(labelsize=12)
    ax[0].legend()

    # normalized
    ax[1].plot(o.index.values,o['p'], color='cyan',label='Omission in interval')    
    ax[1].plot(n.index.values,n['p'], color='gray',label='No omission in interval')      
    ax[1].fill_between(o.index.values,
        o['p']-o['ci'],o['p']+o['ci'],color='lightblue',alpha=.5)
    ax[1].fill_between(n.index.values,
        n['p']-n['ci'],n['p']+n['ci'],color='lightgray',alpha=.5)
    ax[1].set_xlabel('# images between licking bouts',fontsize=16)

    if not cumsum:
        ax[1].set_ylim(0,.25)
        ax[1].set_ylabel('prob',fontsize=16) 
    else:
        ax[1].set_ylabel('cumulative prob',fontsize=16)
    ax[1].set_xlim(1,max_interval)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].xaxis.set_tick_params(labelsize=12)
    ax[1].yaxis.set_tick_params(labelsize=12)
    ax[1].legend()
    plt.tight_layout()


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



