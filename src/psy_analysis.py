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
# bout_df, df = pa.compile_interval_duration(summary_df, version)
# generative_df = pa.demonstrate_sampling_issue(bout_df)
# pa.plot_interval_durations(bout_df);plt.title('data');plt.tight_layout()
# pa.plot_interval_durations(generative_df);plt.title('shuffle');plt.tight_layout()
# pa.plot_cumulative_distribution(bout_df, generative_df)

def demonstrate_sampling_issue(bout_df):
    generative_df = bout_df.copy()
    generative_df['omitted_interval'] = [generate_omission(row.interval_duration) \
        for index, row in generative_df.iterrows()]
    return generative_df
        
def generate_omission(time):

    while time > .75:
        if np.random.rand() < .05:
            return True
        else:
            time = time - .75
    return False

def build_session_interval_df(bsid,version):
    session = pgt.get_data(bsid)
    pm.get_metrics(session)

    df = session.stimulus_presentations
    df['first_lick_time'] = [row.licks[0] if len(row.licks) > 0 else np.nan
        for index, row in df.iterrows()]
    df['last_lick_time'] = [row.licks[-1] if len(row.licks) > 0 else np.nan
        for index, row in df.iterrows()]

    print('saving')
    df.to_csv(pgt.get_directory(version, \
        subdirectory='interval_df')+str(bsid)+'.csv') 

def compute_interval_duration(df):
    
    # Annotate between bouts
    df['bout_number'].fillna(method='ffill',inplace=True)
    df['between_bouts'] = (~df['in_lick_bout']) & \
        (~df['bout_start'])
    df['interval_number'] = df['bout_number']
    df.at[~df['between_bouts'],'interval_number'] = np.nan

    # Annotate interval durations
    df.at[df['between_bouts'],'last_lick_time'] = np.nan
    df.at[df['between_bouts'],'first_lick_time'] = np.nan
    df['first_lick_time'].fillna(method='bfill',inplace=True)
    df['last_lick_time'].fillna(method='ffill',inplace=True)
    df['interval_duration'] = df['first_lick_time'] - df['last_lick_time']
    df.at[~df['between_bouts'],'interval_duration']=np.nan

    # Annotate interval start/stops
    df['interval_start'] = df['bout_end'].shift(1) & ~df['bout_start']
    df['interval_end'] = df['bout_start'].shift(-1) & ~df['bout_end']

    # Annotate if post-interval bout was rewarded. ie, we are "pre" reward
    df['pre_reward'] = df['rewarded']
    df.at[~df['bout_start'],'pre_reward'] = np.nan
    df['pre_reward'].fillna(method='bfill',inplace=True)
    df['pre_reward'] = df['pre_reward'].astype(bool)

    # Annotate if pre-interval bout was rewarde., ie, we are "post" reward
    df['post_reward'] = df['rewarded']
    df.at[~df['bout_start'],'post_reward'] = np.nan
    df['post_reward'].fillna(method='ffill',inplace=True)
    df['post_reward'] = df['post_reward'].astype(bool)

    # Annotate omission times
    df['omission_time'] = df['start_time']-df['last_lick_time']
    df.at[~df['omitted'],'omission_time'] = np.nan
    df.at[~df['between_bouts'],'omission_time'] = np.nan    
 
    # Group into bout intervals
    bout_df = pd.DataFrame()
    g = df.groupby(['interval_number'])
    bout_df['omitted_interval'] = g['omitted'].any()
    bout_df['change_in_interval'] = g['is_change'].any()
    bout_df['pre_reward'] = g['pre_reward'].any()
    bout_df['post_reward'] = g['post_reward'].any()
    bout_df['images_between_bouts'] = g.size()
    bout_df['interval_duration'] = g['interval_duration'].mean()
    bout_df['time_from_last_change'] = g['time_from_last_change'].min() 
    bout_df['omission_time'] = g['omission_time'].min()
    return bout_df,df


def compile_interval_duration(summary_df,version):
    
    dfs = []
    b_dfs = []
    crash = 0
    for index, row in tqdm(summary_df.iterrows(),total=summary_df.shape[0]):
        try:
            strategy_dir = pgt.get_directory(version, subdirectory='interval_df')
            interval_df = pd.read_csv(strategy_dir+str(row.behavior_session_id)+'.csv')
        except Exception as e:
            crash += 1
            print(e)
        else:
            bout_df,df = compute_interval_duration(interval_df)
            bout_df['behavior_session_id'] = row.behavior_session_id 
            b_dfs.append(bout_df)
            dfs.append(df)
    print(crash)

    bout_df =  pd.concat(b_dfs)
    bout_df = bout_df.merge(
        summary_df[['behavior_session_id','visual_strategy_session']], 
        on='behavior_session_id')
    df = pd.concat(dfs)

    return bout_df, df

def plot_cumulative_distribution(bout_df, generative_df, timing_only=True,
    mark_median=True,remove_changes=True,remove_pre_rewards=True,
    remove_post_rewards=True, min_change_delay=10):

    if timing_only:
        bout_df = bout_df.query('not visual_strategy_session').copy()
        generative_df = generative_df.query('not visual_strategy_session').copy()
    if remove_changes:
        bout_df = bout_df.query('not change_in_interval') 
        generative_df = generative_df.query('not change_in_interval') 
    if remove_pre_rewards:
        bout_df = bout_df.query('not pre_reward')
        generative_df = generative_df.query('not pre_reward')
    if remove_post_rewards:
        bout_df = bout_df.query('not post_reward')
        generative_df = generative_df.query('not post_reward')
    bout_df = bout_df.query('time_from_last_change >@min_change_delay')
    generative_df = generative_df.query('time_from_last_change >@min_change_delay')

    plt.figure()
    bins = np.arange(0,20.5,.1)
    plt.hist(bout_df.query('not omitted_interval')['interval_duration'],
        bins=bins,alpha=1,density=True,color='gray',label='data - no omission',
        cumulative=True,histtype='step')
    plt.hist(bout_df.query('omitted_interval')['interval_duration'],
        bins=bins,alpha=1,density=True,color='blue',label='data - omission',
        cumulative=True,histtype='step')
    plt.hist(generative_df.query('not omitted_interval')['interval_duration'],
        bins=bins,alpha=.5,density=True,color='green',label='shuffle - omission',
        cumulative=True,histtype='step')
    plt.hist(generative_df.query('omitted_interval')['interval_duration'],
        bins=bins,alpha=.5,density=True,color='m',label='shuffle - no omission',
        cumulative=True,histtype='step')

    plt.ylabel('cumulative probability of licking',fontsize=16)
    plt.xlabel('inter-bout duration (s)',fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.xlim(0,10)
    plt.ylim(0,1)
    plt.legend(loc='lower right')
    plt.tight_layout()


def plot_interval_durations(bout_df, timing_only=True,mark_median=True,
    remove_changes=True,remove_pre_rewards=True,remove_post_rewards=True,
    min_change_delay=10):

    if timing_only:
        bout_df = bout_df.query('not visual_strategy_session').copy()
    if remove_changes:
        bout_df = bout_df.query('not change_in_interval') 
    if remove_pre_rewards:
        bout_df = bout_df.query('not pre_reward')
    if remove_post_rewards:
        bout_df = bout_df.query('not post_reward')
    bout_df = bout_df.query('time_from_last_change >@min_change_delay')
    
    plt.figure()
    bins = np.arange(0,20.5,.5)
    plt.hist(bout_df.query('not omitted_interval')['interval_duration'],
        bins=bins,alpha=.5,density=True,color='gray',label='no omission')
    plt.hist(bout_df.query('omitted_interval')['interval_duration'],
        bins=bins,alpha=.5,density=True,color='lightblue',label='omission')

    if mark_median:
        medians = bout_df.groupby('omitted_interval')['interval_duration'].median()
        ylims = plt.ylim()

        plt.plot(medians.loc[False],ylims[1],'v',color='gray')
        plt.plot(medians.loc[True],ylims[1],'v',color='lightblue')

    plt.ylabel('probability',fontsize=16)
    plt.xlabel('inter-bout duration (s)',fontsize=16)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    plt.xlim(0,20)
    plt.ylim(top=.38)
    plt.legend()
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



