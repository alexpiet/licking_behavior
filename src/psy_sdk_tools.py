import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import psy_tools as ps
import copy
from allensdk.brain_observatory.behavior.swdb import behavior_project_cache as bpc
import allensdk.brain_observatory.behavior.swdb.utilities as tools


def build_response_latency(cdf):
    '''
        Estimates the response_latency for each flash by taking the time of the first lick. 
        WARNING: This may differ from the response_latency in the trials table. This function was meant as a temporary fix 
        
        INPUT:  
        cdf, the cluster dataframe. 
    
        OUTPUT:
        cdf, with a new column 'response_latency', which is the time in seconds to the first lick after image onset, and is NaN is no licks were registered.
    '''
    response_latency = cdf['licks'] -cdf['start_time']
    response_latency = response_latency.reset_index()
    response_latency['latency'] = np.nan
    response_latency['latency'] = response_latency[0].str[0]
    response_latency = response_latency.set_index('index')
    cdf['response_latency'] = response_latency['latency']
    return cdf

def running_behavior_by_cluster(cdf,cluster_num,session=None,stage=""):
    '''
        Plots the Average running speed for each flash by behavioral cluster. 
        
        INPUTS:
        cdf, the cluster dataframe
        cluster_num, which cluster size to use as a string, like '3'
        session, the session ID, just used for plotting the title
        stage, the session stage as a str, just used for plotting the title
    '''
    fig,ax = plt.subplots(nrows=1,ncols=5,sharey=True,figsize=(16,5))
    sns.barplot(x=cluster_num, y="running_speed", data=cdf[cdf[cluster_num] != -1],ax=ax[0])
    sns.barplot(x=cluster_num, y="running_speed", data=cdf[cdf[cluster_num] != -1].query('change==True'),ax=ax[1])
    sns.barplot(x=cluster_num, y="running_speed", data=cdf[cdf[cluster_num] != -1].query('omitted==True'),ax=ax[2])
    sns.barplot(x=cluster_num, y="running_speed", data=cdf[(cdf[cluster_num] != -1) & (cdf.licks.str.len() > 0)],ax=ax[3])
    sns.barplot(x=cluster_num, y="running_speed", data=cdf[(cdf[cluster_num] != -1) & (cdf.rewards.str.len() > 0)],ax=ax[4])
    ax[0].title.set_text('All images')
    ax[1].title.set_text('change images')
    ax[2].title.set_text('omitted images')
    ax[3].title.set_text('Mouse licked')
    ax[4].title.set_text('Hit')
    rows = ['Running Speed']
    pad = 5
    for axs, row in zip(ax[:], rows):
        axs.annotate(row, xy=(0, 0.5), xytext=(-axs.yaxis.labelpad - pad, 0),
                    xycoords=axs.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')

    plt.gcf().suptitle("Session: "+str(session)+" "+stage)

def latency_behavior_by_cluster(cdf,cluster_num,session=None,stage=""):
    '''
        Plots the Response Latency by behavioral cluster
        
        INPUTS:
        cdf, the cluster dataframe
        cluster_num, which cluster size to use as a string, like '3'
        session, the session ID, just used for plotting the title
        stage, the session stage as a str, just used for plotting the title       
    
    '''
    num_clusters = copy.copy(cluster_num)
    cluster_num = str(cluster_num)
    fig,ax = plt.subplots(nrows=1,ncols=5,sharey=True,figsize=(16,5))
    sns.barplot(x=cluster_num, y="response_latency", data=cdf[cdf[cluster_num] != -1],ax=ax[0])
    sns.barplot(x=cluster_num, y="response_latency", data=cdf[cdf[cluster_num] != -1].query('omitted==True'),ax=ax[1])
    sns.barplot(x=cluster_num, y="response_latency", data=cdf[(cdf[cluster_num] != -1) & (cdf.rewards.str.len() > 0)].query('change==True'),ax=ax[2])
    sns.barplot(x=cluster_num, y="response_latency", data=cdf[(cdf[cluster_num] != -1) & (cdf.rewards.str.len() == 0)].query('change==False'),ax=ax[3])
    num_responses = []
    nr_df = pd.DataFrame()
    for i in range(0,num_clusters):    
        num_responses.append(np.sum(~np.isnan(cdf[cdf[cluster_num] == i].response_latency))/len(cdf[cdf[cluster_num]==i]))
        nr_df.loc[i,'fraction_responses'] = num_responses[-1]
        nr_df.loc[i,cluster_num] = i
    sns.barplot(x=cluster_num, y="fraction_responses", data=nr_df,ax=ax[4])
    ax[0].title.set_text('All images')
    ax[1].title.set_text('omitted images')
    ax[2].title.set_text('Hit')
    ax[3].title.set_text('False Alarm')
    ax[4].title.set_text('Fraction responded')
    rows = ['Response Latency']
    pad = 5
    for axs, row in zip(ax[:], rows):
        axs.annotate(row, xy=(0, 0.5), xytext=(-axs.yaxis.labelpad - pad, 0),
                    xycoords=axs.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    plt.gcf().suptitle("Session: "+str(session)+" "+stage)

def mean_response_by_cluster(cdf,cluster_num,session=None,stage=""):
    '''
        Plots the mean_response by behavioral cluster. Does not do any normalization by cell or session
        
        INPUTS:
        cdf, the cluster dataframe
        cluster_num, which cluster size to use as a string, like '3'
        session, the session ID, just used for plotting the title
        stage, the session stage as a str, just used for plotting the title       
    
    '''
    fig,ax = plt.subplots(nrows=2,ncols=4,sharey=True,figsize=(16,8))
    sns.barplot(x = cluster_num, y="mean_response",data=cdf[cdf[cluster_num] != -1],ax =ax[0,0])
    sns.barplot(x = cluster_num, y="mean_response",data=cdf[cdf[cluster_num] != -1].query('change==True'),ax =ax[0,1])
    sns.barplot(x = cluster_num, y="mean_response",data=cdf[cdf[cluster_num] != -1].query('pref_stim==True'),ax =ax[0,2])
    sns.barplot(x = cluster_num, y="mean_response",data=cdf[cdf[cluster_num] != -1].query('pref_stim==True').query('change==True'),ax =ax[0,3])
    sns.barplot(x = cluster_num, y="mean_response",data=cdf[cdf[cluster_num] != -1].query('p_value < 0.05'),ax =ax[1,0])
    sns.barplot(x = cluster_num, y="mean_response",data=cdf[cdf[cluster_num] != -1].query('p_value < 0.05').query('change==True'),ax =ax[1,1])
    sns.barplot(x = cluster_num, y="mean_response",data=cdf[cdf[cluster_num] != -1].query('p_value < 0.05').query('pref_stim==True'),ax =ax[1,2])
    sns.barplot(x = cluster_num, y="mean_response",data=cdf[cdf[cluster_num] != -1].query('p_value < 0.05').query('pref_stim==True').query('change==True'),ax =ax[1,3])
    ax[0,0].title.set_text('all images')
    ax[0,1].title.set_text('change images')
    ax[0,2].title.set_text('preferred images')
    ax[0,3].title.set_text('change+preferred images')
    ax[1,0].title.set_text('all images')
    ax[1,1].title.set_text('change images')
    ax[1,2].title.set_text('preferred images')
    ax[1,3].title.set_text('change+preferred images')
    rows = ['all','p < 0.05']
    pad = 5
    for axs, row in zip(ax[:,0], rows):
        axs.annotate(row, xy=(0, 0.5), xytext=(-axs.yaxis.labelpad - pad, 0),
                    xycoords=axs.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center')
    plt.gcf().suptitle("Session: "+str(session)+" "+stage)

def get_joint_table(fit,session, use_all_clusters=True,passive=False):
    '''
        Creates and returns the joint dataframe by merging the flash_response_df and the cluster labels
        
        INPUTS:
        fit, the psy-track model fit
        session, the SDK session object
        use_all_clusters, if TRUE uses the clusters defined over all sessions, otherwise uses the individual session clusters
        passive, set to TRUE if this is a passive session

        OUTPUT:
        cdf, the cluster dataframe, which is the flash_response_df with several new columns added:
           'included'           A boolean column of whether that flash was included in the psytrack model
           '2'                  Cluster labels for 2 cluster segmenting
           '3'                  Cluster labels for 3 cluster segmenting
           '4'                  Cluster labels for 4 cluster segmenting
           'response_latency'   Estimated behavioral response latency
    '''
    fr = session.flash_response_df
    df = build_cluster_table(session,fit,fr,use_all_clusters = use_all_clusters,passive=passive)   
    cdf = build_joint_table(fr,df)
    cdf = build_response_latency(cdf)
    return cdf

def build_joint_table(fr,df):
    '''
        Merges the flash_response_df, and the cluster_df
        
        INPUTS:
        fr, the SDK flash_response_df
        df, the cluster dataframe from build_cluster_table
        
        OUTPUTS:
        cdf, the cluster combined dataframe
    '''
    fr = fr.reset_index().set_index('flash_id')
    combined = pd.merge(fr,df,how='outer',left_index=True, right_index=True)
    return combined

def build_cluster_table(session, fit,fr,use_all_clusters = True,passive=False):
    '''
        Builds the cluster dataframe
        
        INPUTS:
        session, the SDK session object
        fit, the psytrack model fit
        fr, the flash_response_df
        use_all_clusters, if TRUE uses the clusters defined over all behavioral sessions
        passive, set to TRUE if this is a passive session
    
        OUTPUTS:
        the cluster_dataframe
    '''
    flash_ids = fr['flash_id'].unique()
    df, included, not_included = get_flash_ids(session,flash_ids)
    df['2'] = -1
    df['3'] = -1
    df['4'] = -1
    
    if passive:
        df.at[df['included'],'2'] = 2
        df.at[df['included'],'3'] = 3
        df.at[df['included'],'4'] = 4
    else:
        if use_all_clusters:
            df.at[df['included'],'2'] = fit['all_clusters']['2'][1]
            df.at[df['included'],'3'] = fit['all_clusters']['3'][1]
            df.at[df['included'],'4'] = fit['all_clusters']['4'][1]   
        else:
            df.at[df['included'],'2'] = fit['clusters']['2'][1]
            df.at[df['included'],'3'] = fit['clusters']['3'][1]
            df.at[df['included'],'4'] = fit['clusters']['4'][1]
    return df

def get_flash_ids(session,flash_ids):
    '''
        Estimates which flashes were used in the psy-track model. This function is really hacky and frequently crashes. Newer model fits will include the list of flash_ids in the fit object, so hopefully this function goes way soon.
        
        INPUTS:
        session, the SDK session object
        flash_ids, the list of all possible flash_ids

        OUTPUTS:
        df, a dataframe with boolean column 'included'
        included, a list of included flash_ids
        not_included, a list of not included flash_ids
    
    '''
    included = []
    not_included = []
    df = pd.DataFrame(index=flash_ids)
    df['included'] = False 
    for index, row in session.stimulus_presentations.iterrows():
        start_time = row.start_time
        if (not check_grace_windows(session, start_time)):
            included.append(index)
            df.at[index,'included'] = True
        else:
            not_included.append(index)           
    return df, included, not_included
   

def check_grace_windows(session,time_point):
    '''
        Returns true if the time point is inside the grace period after reward delivery from an earned reward or auto-reward
    '''
    hit_end_times = session.trials.stop_time[session.trials.hit].values
    hit_response_time = session.trials.response_latency[session.trials.hit].values + session.trials.change_time[session.trials.hit].values
    inside_grace_window = np.any((hit_response_time < time_point ) & (hit_end_times > time_point))
    
    auto_reward_time = session.trials.change_time[(session.trials.auto_rewarded) & (~session.trials.aborted)] + .5
    auto_end_time = session.trials.stop_time[(session.trials.auto_rewarded) & (~session.trials.aborted)]
    inside_auto_window = np.any((auto_reward_time < time_point) & (auto_end_time > time_point))
    return inside_grace_window | inside_auto_window

def build_multi_session_joint_table(ids,cache, manifest, use_all_clusters=True):
    '''
        Merges several sessions into one cluster dataframes
        
        INPUTS:
        ids, a list of behavioral session ids
        cache, the SDK cache
        manifest, the manifest of all sessions
        use_all_clusters, if TRUE uses the cluster labels from all behavioral sessions 

        OUTPUTS:
        sessions, the SDK session objects for each ID
        fits, the model fits for each ID
        mega_cdf, the cluster dataframe jointly for all flashes on all sessions 
        
    '''
    this_manifest = manifest.set_index('ophys_experiment_id').loc[ids]
    sessions = []
    fits=[]
    cdfs=[]
    for id in ids:
        print(id)
        try:
            passive = manifest[manifest['ophys_experiment_id'] == id]['stage_name'].str[-1].values[0] == "e"
            if not passive:
                fit = ps.load_fit(id)
            else:
                print("  passive")
                fit = None
            session = cache.get_session(id)
            fits.append(fit)
            sessions.append(session)
            cdfs.append(get_joint_table(fit,session,passive=passive))
        except:
            print("  crash")
    mega_cdf = pd.concat(cdfs)
    return sessions,fits,mega_cdf

