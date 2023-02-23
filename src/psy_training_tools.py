import os
import numpy as np
import pandas as pd
import psy_tools as ps
import psy_output_tools as po
import psy_general_tools as pgt
import matplotlib.pyplot as plt
from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache
plt.ion()

BEHAVIOR_VERSION = 21

def dev_notes():
    # Get a list of training sessions 
    train_manifest = get_training_manifest()

    # Get inventory of what sessions have been fit
    inventory = get_training_inventory()

    # Build a summary table with fit information
    train_summary = build_training_summary_table(version)

 
    # outdated below here
    # Train Summary is a dataframe with model fit information
    train_summary = po.get_training_summary_table(version)
    ophys_summary = po.get_ophys_summary_table(version)
    mouse_summary = po.get_mouse_summary_table(version)
    full_table = get_full_behavior_table(train_summary, ophys_summary)
    full_table_no_lapse = get_full_behavior_table(train_summary, ophys_summary,filter_lapsed=True)
    
    # Plot Averages by training stage 
    plot_average_by_stage(full_table, metric='strategy_dropout_index')
    plot_all_averages_by_stage(full_table,version)
    plot_all_averages_by_stage(full_table,version,plot_mouse_groups=True)
    plot_all_averages_by_stage(full_table,version,plot_each_mouse=True)
    plot_all_averages_by_stage(full_table,version,plot_cre=True)
    
    # Plot Average by Training session
    plot_all_averages_by_day(full_table, mouse_summary, version)
    plot_all_averages_by_day_mouse_groups(full_table, mouse_summary, version)
    plot_all_averages_by_day_cre(full_table, mouse_summary, version)
    
    # SAC plot
    training = po.get_training_summary_table(20)
    skip = ['OPHYS_1','OPHYS_3','OPHYS_4','OPHYS_6','OPHYS_0_habituation','TRAINING_5_lapsed','TRAINING_4_lapsed']
    plot_average_by_stage(training, metric='num_hits',filetype='_sac.png',version=20,alpha=1,SAC=True, metric_name='# Hits / Session',skip=skip)
    
    # TODO, Broken, Issue #92
    pv.plot_task_timing_by_training_duration(summary_df,version)


def get_training_manifest(non_ophys=True):
    '''
        Return a table of all training/ophys sessions
        Removes mice without ophys session fits

        non_ophys, if True (default) removes ophys sessions 
    '''
    cache_dir = r'//allen/programs/braintv/workgroups/nc-ophys/visual_behavior/platform_paper_cache'
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=cache_dir)
    training = cache.get_behavior_session_table()  

    # Sort by BSID 
    training.sort_index(inplace=True)

    # Move BSID to column
    training = training.reset_index()

    # Remove passive sessions
    training.dropna(subset=['session_type'],inplace=True)
    training['passive'] = ['passive' in x for x in training.session_type]
    training = training.query('not passive').drop(columns=['passive']).copy()

    # Mark ophys sessions
    training['ophys'] = [('OPHYS' in x)and('habituation' not in x)\
        for x in training.session_type]

    # Calculate pre ophys number, and training number
    training['pre_ophys_number'] = training.groupby(['mouse_id','ophys'])\
        .cumcount(ascending=False)+1
    training['training_number'] = training.groupby(['mouse_id'])\
        .cumcount(ascending=True)+1
    training['tmp'] = training.groupby(['mouse_id','ophys']).cumcount()
    training.loc[training['ophys'],'pre_ophys_number'] = \
        -training[training['ophys']]['tmp']
    training= training.drop(columns=['tmp'])

    # Remove sessions without ophys fits
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    mice_id = summary_df['mouse_id'].unique()
    training = training.query('mouse_id in @mice_id').copy()

    # Remove OPHYS sessions
    if non_ophys:
        training = training.query('not ophys').copy()

    return training

def get_training_inventory(version=None):

    if version is None:
        version = BEHAVIOR_VERSION

    manifest = get_training_manifest()

    # Check what is actually available. 
    fit_directory=pgt.get_directory(version,subdirectory='fits')
    df_directory=pgt.get_directory(version,subdirectory='strategy_df') 
    for index, row in manifest.iterrows():
        fit_filename = fit_directory + str(row.behavior_session_id) + ".pkl"         
        manifest.at[index, 'behavior_fit_available'] = os.path.exists(fit_filename)

        summary_filename = df_directory+ str(row.behavior_session_id)+'.csv'
        manifest.at[index, 'strategy_df_available'] = os.path.exists(summary_filename)

    # Summarize inventory for this model version
    inventory = {}    
    inventory['fit_sessions'] = \
        manifest.query('behavior_fit_available == True')['behavior_session_id']
    inventory['missing_sessions'] = \
        manifest.query('behavior_fit_available != True')['behavior_session_id']
    inventory['with_strategy_df'] = \
        manifest.query('strategy_df_available == True')['behavior_session_id']
    inventory['without_strategy_df'] = \
        manifest.query('strategy_df_available != True')['behavior_session_id']
    inventory['num_sessions'] = len(manifest)
    inventory['num_fit'] = len(inventory['fit_sessions'])
    inventory['num_missing'] = len(inventory['missing_sessions'])
    inventory['num_with_strategy_df'] = len(inventory['with_strategy_df'])
    inventory['num_without_strategy_df'] = len(inventory['without_strategy_df'])
    inventory['version'] = version

    print('Number of sessions fit:     {}'.format(inventory['num_fit']))
    print('Number of sessions missing: {}'.format(inventory['num_missing']))
    print('Number of strategy missing: {}'.format(inventory['num_without_strategy_df']))
    return inventory



def get_training_summary_table(version):
    model_dir = pgt.get_directory(version,subdirectory='summary')
    return pd.read_pickle(model_dir+'_training_summary_table.pkl')


def build_training_summary_table(version):
    ''' 
        Saves out the training table as a csv file 
    '''
    # Build core table
    print('Building training summary table')
    print('Loading model fits')
    training_summary = build_core_training_table(version)
    training_summary = po.build_strategy_labels(training_summary)

    print('Loading image by image information')
    training_summary = add_time_aligned_training_info(summary_df, version)

    print('Adding engagement information')
    training_summary = add_training_engagement_metrics(summary_df)
    
    print('Savining')
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    training_summary.to_pickle(model_dir+'_training_summary_table.pkl')

    return training_summary

def build_core_training_table(version):
    train_summary = get_training_manifest() 
    print('WARNING, core training table not yet implemented')
    return training_summary

def add_time_aligned_training_info(training_summary, version):
    print('WARNING, time aligned training info not added yet')
    return training_summary

def add_training_engagement_metrics(training_summary):
    print('WARNING, training engagement not added yet')
    return training_summary


## Training functions below here, in development
################################# 

def build_pseudo_stimulus_presentations(session):#TODO, Issue #92
    '''
        For Training 0/1 the stimulus was not images but presented serially. This
        function builds a pseudo table of stimuli by breaking up the continuously
        presented stimuli into repeated stimuli. This is just to make the behavior model
        fit. 
    '''
    raise Exception('Need to update')
    # Store the original
    session.stimulus_presentations_sdk = session.stimulus_presentations.copy()

    # Get the basic data frame by iterating start times
    session.stimulus_presentations = pd.DataFrame()
    start_times = []
    image_index = []
    image_name =[]
    for index, row in session.stimulus_presentations_sdk.iterrows():
        new_images = list(np.arange(row['start_time'],row['stop_time'],0.75))
        start_times = start_times+ new_images
        image_index = image_index + [row['image_index']]*len(new_images) 
        image_name = image_name + [row['image_name']]*len(new_images) 
    session.stimulus_presentations['start_time'] = start_times
    session.stimulus_presentations['image_index'] = image_index
    session.stimulus_presentations['image_name'] = image_name

    # Filter out very short stimuli which happen because the stimulus duration was not
    # constrainted to be a multiple of 750ms
    session.stimulus_presentations['duration'] = \
        session.stimulus_presentations.shift(-1)['start_time']\
        -session.stimulus_presentations['start_time']
    session.stimulus_presentations = \
        session.stimulus_presentations.query('duration > .25').copy().reset_index()
    session.stimulus_presentations['duration'] = \
        session.stimulus_presentations.shift(-1)['start_time']\
        -session.stimulus_presentations['start_time']


    # Add other columns
    session.stimulus_presentations['omitted'] = False
    session.stimulus_presentations['stop_time'] =\
        session.stimulus_presentations['duration']\
        +session.stimulus_presentations['start_time']
    session.stimulus_presentations['image_set'] = \
        session.stimulus_presentations_sdk.iloc[0]['image_set']

    return session

def training_add_licks_each_image(stimulus_presentations, licks):#TODO, Issue #92
    raise Exception('Need to update')
    lick_times = licks['timestamps'].values
    licks_each_image = stimulus_presentations.apply(
        lambda row: lick_times[((lick_times > row["start_time"]) \
        & (lick_times < row["stop_time"]))],
        axis=1)
    stimulus_presentations['licks'] = licks_each_image
    return stimulus_presentations

def training_add_rewards_each_image(stimulus_presentations,rewards):#TODO, Issue #92
    raise Exception('Need to update')
    reward_times = rewards['timestamps'].values
    rewards_each_image = stimulus_presentations.apply(
        lambda row: reward_times[((reward_times > row["start_time"]) \
        & (reward_times < row["stop_time"]))],
        axis=1,
    )
    stimulus_presentations['rewards'] = rewards_each_image
    return stimulus_presentations





def plot_all_averages_by_stage(full_table, version,filetype='.svg',plot_each_mouse=False, plot_mouse_groups=False,plot_cre=False):
    if plot_each_mouse or plot_mouse_groups or plot_cre:
        mouse = po.get_mouse_summary_table(version)
    else:
        mouse=None
    plot_average_by_stage(full_table,metric='strategy_dropout_index',version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse,plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='visual_only_dropout_index',version=version,flip_axis=True,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse,plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='timing_only_dropout_index',version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse,plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='strategy_weight_index', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='avg_weight_task0', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='avg_weight_bias', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='avg_weight_timing1D', version=version,flip_axis=True,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='lick_hit_fraction', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='num_hits', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='num_fa', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='num_cr', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='num_miss', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='num_aborts', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='session_roc', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)
    plot_average_by_stage(full_table,metric='fraction_engaged', version=version,filetype=filetype,mouse=mouse,plot_each_mouse=plot_each_mouse, plot_mouse_groups=plot_mouse_groups,plot_cre=plot_cre)

def plot_average_by_stage_inner(group,color='k',label=None,skip=[],alpha=.2):
    group['std_err'] = group['std']/np.sqrt(group['count'])
    for index, row in group.iterrows():
        if (index not in skip) & (index[1:] not in skip):
            if index in ['TRAINING_2','TRAINING_3','TRAINING_4_handoff', 'TRAINING_5_handoff','_OPHYS_1','_OPHYS_3','_OPHYS_4','_OPHYS_6','_OPHYS_0_habituation']:
                plt.plot(row['mean'],index,'o',zorder=3,color=color)
            else:       
                plt.plot(row['mean'],index,'o',color=color,alpha=alpha,zorder=3)
            plt.plot([row['mean']-row['std_err'], row['mean']+row['std_err']],[index, index], '-',alpha=alpha,zorder=2,color=color)
            if index == 'TRAINING_2':
                plt.plot(row['mean'],index,'o',zorder=3,color=color,label=label)

def plot_average_by_stage(full_table,ophys=None,metric='strategy_dropout_index',savefig=True,version=None,flip_axis = False,filetype='.png',plot_each_mouse=False,mouse=None, plot_mouse_groups=False,plot_cre=False,skip=[],alpha=.2,SAC=False,metric_name=''):
    
    full_table['clean_session_type'] = [clean_session_type(x) for x in full_table.session_type]

    plt.figure(figsize=(6.5,3.75))
    if (not plot_mouse_groups) & (not plot_cre):
        # Plot average across all groups
        group = full_table.groupby('clean_session_type')[metric].describe()
        plot_average_by_stage_inner(group,skip=skip,alpha=alpha)
    elif plot_mouse_groups:
        cmap = plt.get_cmap('plasma')
        # Plot Visual Mice
        visual_color = cmap(225)
        visual_mice = mouse.query('strategy == "visual"').index.values
        visual = full_table.query('donor_id in @visual_mice').copy()
        group = visual.groupby('clean_session_type')[metric].describe()
        plot_average_by_stage_inner(group,color=visual_color,label='Visual Ophys Mice',skip=skip,alpha=alpha)

        # Plot Timing Mice
        timing_color = cmap(0)
        timing_mice = mouse.query('strategy == "timing"').index.values
        timing = full_table.query('donor_id in @timing_mice').copy()
        group = timing.groupby('clean_session_type')[metric].describe()
        plot_average_by_stage_inner(group,color=timing_color,label='Timing Ophys Mice',skip=skip,alpha=alpha)
    else:
        # plot cre lines
        sst_color = (158/255,218/255,229/255)
        vip_color = (197/255,176/255,213/255)
        slc_color = (255/255,152/255,150/255)
        sst_mice = mouse.query('cre_line == "Sst-IRES-Cre"').copy()
        vip_mice = mouse.query('cre_line == "Vip-IRES-Cre"').copy()
        slc_mice = mouse.query('cre_line == "Slc17a7-IRES2-Cre"').copy()
        sst_mice_ids = sst_mice.index.values
        vip_mice_ids = vip_mice.index.values
        slc_mice_ids = slc_mice.index.values
        sst = full_table.query('donor_id in @sst_mice_ids').copy()
        vip = full_table.query('donor_id in @vip_mice_ids').copy()
        slc = full_table.query('donor_id in @slc_mice_ids').copy()
        group = vip.groupby('clean_session_type')[metric].describe()
        plot_average_by_stage_inner(group,color=vip_color,label='Vip',skip=skip,alpha=alpha)
        group = sst.groupby('clean_session_type')[metric].describe()
        plot_average_by_stage_inner(group,color=sst_color,label='Sst',skip=skip,alpha=alpha)
        group = slc.groupby('clean_session_type')[metric].describe()
        plot_average_by_stage_inner(group,color=slc_color,label='Slc',skip=skip,alpha=alpha)

    # Clean up plot
    if flip_axis:
        plt.gca().invert_xaxis()

    labels = [x[1:] if x[0] == "_" else x for x in group.index.values]
    labels = [x for x in labels if x not in skip]
    if not SAC: 
        plt.gca().set_yticks(np.arange(0,len(labels)))
        plt.gca().set_yticklabels(labels,rotation=0)   
        plt.axvline(0,color='k',linestyle='--',alpha=.5)
        plt.axhline(9.5, color='k',linestyle='--', alpha=.5)
        plt.xlabel(metric)
    else:
        plt.xlim(0,125)
        plt.gca().set_yticks(np.arange(0,len(labels)))
        plt.gca().set_yticklabels(labels,rotation=0,fontsize=14)   
        plt.xlabel(metric_name,fontsize=14)   
        plt.gca().tick_params(axis='x',labelsize=14) 
    
    if plot_mouse_groups or plot_cre:
        plt.legend()
    if metric =='session_roc':
        plt.xlim([.6,1])

    if plot_each_mouse:
        cmap = plt.get_cmap('plasma')
        norm = plt.Normalize(np.min(mouse['strategy_dropout_index']), np.max(mouse['strategy_dropout_index']))
        mouse_ids = mouse.index.values
        for mouse_id in mouse_ids:
            mouse_avg = mouse.loc[mouse_id].strategy_dropout_index
            mouse_table = full_table.query('donor_id == @mouse_id').copy()
            group = mouse_table.groupby('clean_session_type')[metric].describe()
            plt.plot(group['mean'],group.index,'-', alpha=.3,zorder=1,color=cmap(norm(mouse_avg)))

    if ophys is not None:
        ophys['clean_session_type'] = [clean_session_type(x) for x in ophys.session_type]
        group = ophys.groupby('clean_session_type')[metric].describe()
        group['std_err'] = group['std']/np.sqrt(group['count'])
        for index, row in group.iterrows():
            plt.plot(row['mean'],index,'bo')
            plt.plot([row['mean']-row['std_err'], row['mean']+row['std_err']],[index, index], 'b-')

    plt.tight_layout()
    if savefig:
        directory = pgt.get_directory(version)
        if plot_each_mouse:
            plt.savefig(directory+'figures_training/mouse_'+metric+'_by_stage'+filetype) 
        elif plot_mouse_groups:
            plt.savefig(directory+'figures_training/mouse_groups_'+metric+'_by_stage'+filetype)
        elif plot_cre:
            plt.savefig(directory+'figures_training/cre_'+metric+'_by_stage'+filetype)
        else:
            plt.savefig(directory+'figures_training/avg_'+metric+'_by_stage'+filetype)

def clean_session_type(session_type):
    sessions = {
    "OPHYS_0_images_A_habituation":      "_OPHYS_0_habituation",
    "OPHYS_0_images_B_habituation":      "_OPHYS_0_habituation",
    "OPHYS_1_images_A":                  "_OPHYS_1",
    "OPHYS_1_images_B":                  "_OPHYS_1",
    "OPHYS_3_images_A":                  "_OPHYS_3",
    "OPHYS_3_images_B":                  "_OPHYS_3",
    "OPHYS_4_images_A":                  "_OPHYS_4",
    "OPHYS_4_images_B":                  "_OPHYS_4",
    "OPHYS_6_images_A":                  "_OPHYS_6",
    "OPHYS_6_images_B":                  "_OPHYS_6",
    "TRAINING_0_gratings_autorewards_15min":"TRAINING_0",
    "TRAINING_1_gratings":               "TRAINING_1",
    "TRAINING_2_gratings_flashed":       "TRAINING_2",
    "TRAINING_3_images_A_10uL_reward":   "TRAINING_3",
    "TRAINING_3_images_B_10uL_reward":   "TRAINING_3",
    "TRAINING_4_images_A_handoff_lapsed":"TRAINING_4_lapsed",
    "TRAINING_4_images_B_handoff_lapsed":"TRAINING_4_lapsed",
    "TRAINING_4_images_A_handoff_ready": "TRAINING_4_handoff",
    "TRAINING_4_images_B_handoff_ready": "TRAINING_4_handoff",
    "TRAINING_4_images_A_training":      "TRAINING_4",
    "TRAINING_4_images_B_training":      "TRAINING_4",
    "TRAINING_5_images_A_handoff_lapsed":"TRAINING_5_lapsed",
    "TRAINING_5_images_B_handoff_lapsed":"TRAINING_5_lapsed",
    "TRAINING_5_images_A_handoff_ready": "TRAINING_5_handoff",
    "TRAINING_5_images_B_handoff_ready": "TRAINING_5_handoff",
    "TRAINING_5_images_A_training":      "TRAINING_5",
    "TRAINING_5_images_B_training":      "TRAINING_5",
    "TRAINING_5_images_A_epilogue":      "TRAINING_5",
    "TRAINING_5_images_B_epilogue":      "TRAINING_5"
    }
    return sessions[session_type]

def get_full_behavior_table(train_summary, ophys_summary,filter_0=False, filter_1=False,filter_lapsed=False):

    ophys_summary = ophys_summary.copy()
    ophys_summary['pre_ophys_number'] = -ophys_summary.groupby(['donor_id']).cumcount(ascending=True)
   
    if filter_0:
        train_summary = train_summary[~(train_summary.session_type == 'TRAINING_0_gratings_autorewards_15min')]
    if filter_1:
        train_summary = train_summary[~(train_summary.session_type == 'TRAINING_1_gratings')] 
    if filter_lapsed:
        train_summary = train_summary[~train_summary.session_type.isin([
            'TRAINING_4_images_A_handoff_lapsed',           
            'TRAINING_5_images_A_handoff_lapsed',           
            'TRAINING_5_images_B_handoff_lapsed',           
            ])] 

    full_table = train_summary.query('pre_ophys_number > 0').copy()
    full_table = full_table.append(ophys_summary,sort=False)
    full_table = full_table.sort_values(by='behavior_session_id').reset_index(drop=True)
    return full_table 


def get_mouse_pivot_table(train_summary, mouse_summary, metric='strategy_dropout_index'):
    mouse_pivot = train_summary.pivot(index='donor_id',columns='pre_ophys_number',values=[metric]).copy()
    mouse_pivot['ophys_index'] = mouse_summary[metric]
    return mouse_pivot

def plot_mouse_strategy_correlation(train_summary,mouse_summary,version, group_label='',metric='strategy_dropout_index'):
    '''
        Plots each mouse's difference in strategy from its final strategy. 
    '''

    mouse_pivot = get_mouse_pivot_table(train_summary, mouse_summary, metric=metric)

    # Plot each mouse's trajectory    
    plt.figure(figsize=(10,5))
    plt.axvspan(0,6,color='k',alpha=.1)
    plt.axhline(0, color='k',linestyle='--',alpha=0.5)
    xvals = -np.sort(train_summary.pre_ophys_number.unique())
    for dex, mouse in enumerate(range(0,len(mouse_pivot))):
        plt.plot(xvals, mouse_pivot[metric][:].iloc[dex].values- mouse_pivot['ophys_index'].iloc[dex],'o-',alpha=.1)

    # Plot the mean trajectory
    means = []
    for dex,val in enumerate(np.sort(train_summary.pre_ophys_number.unique())):
        means.append(np.nanmean(mouse_pivot[metric][val].values - mouse_pivot['ophys_index'].values))
        plt.plot(-val, np.nanmean(mouse_pivot[metric][val].values - mouse_pivot['ophys_index'].values),'rx')
    plt.plot(xvals, means, 'r-',linewidth=2) 
    plt.xlabel('Sessions before Ophys Stage 1', fontsize=16)  
    plt.xlim(right=6)

    # Save and cleanup
    directory=pgt.get_directory(version)
    if metric is not 'strategy_dropout_index':
        plt.ylabel('Diff in '+metric,fontsize=16)
    else: 
        plt.ylim(-25,25)  
        plt.ylabel('Diff in Strategy Index',fontsize=16)
    plt.savefig(directory+'figures_training/mouse_correlation_'+metric+group_label+'.svg')
    plt.savefig(directory+'figures_training/mouse_correlation_'+metric+group_label+'.png')


def plot_all_averages_by_day(full_table, mouse_summary, version):
    plot_average_by_day(full_table, mouse_summary,version, metric='strategy_dropout_index')    
    plot_average_by_day(full_table, mouse_summary,version, metric='visual_only_dropout_index')    
    plot_average_by_day(full_table, mouse_summary,version, metric='timing_only_dropout_index')    
    plot_average_by_day(full_table, mouse_summary,version, metric='strategy_weight_index')    
    plot_average_by_day(full_table, mouse_summary,version, metric='avg_weight_task0')    
    plot_average_by_day(full_table, mouse_summary,version, metric='avg_weight_timing1D')    
    plot_average_by_day(full_table, mouse_summary,version, metric='avg_weight_bias')
    plot_average_by_day(full_table, mouse_summary,version, metric='lick_hit_fraction')    
    plot_average_by_day(full_table, mouse_summary,version, metric='num_hits')  
    plot_average_by_day(full_table, mouse_summary,version, metric='session_roc')   
    plot_average_by_day(full_table, mouse_summary,version, metric='fraction_engaged')  

def plot_all_averages_by_day_mouse_groups(full_table, mouse_summary, version):
    cmap = plt.get_cmap('plasma')
    visual_color = cmap(225)
    timing_color = cmap(0)
    visual_mice = mouse_summary.query('strategy == "visual"').index.values
    timing_mice = mouse_summary.query('strategy == "timing"').index.values
    visual = full_table.query('donor_id in @visual_mice').copy()
    timing = full_table.query('donor_id in @timing_mice').copy()

    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='strategy_dropout_index',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='strategy_dropout_index',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='visual_only_dropout_index',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='visual_only_dropout_index',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='timing_only_dropout_index',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='timing_only_dropout_index',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='strategy_weight_index',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='strategy_weight_index',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='avg_weight_task0',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='avg_weight_task0',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='avg_weight_timing1D',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='avg_weight_timing1D',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='avg_weight_bias',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='avg_weight_bias',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='lick_hit_fraction',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='lick_hit_fraction',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='num_hits',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='num_hits',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='session_roc',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='session_roc',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')
    plot_average_by_day(visual, mouse_summary,version,label='Visual ophys mice', metric='fraction_engaged',color=visual_color,group_label='_mouse_groups')    
    plot_average_by_day(timing, mouse_summary,version,label='Timing ophys mice', metric='fraction_engaged',fig=plt.gcf(),color=timing_color,group_label='_mouse_groups')

def plot_all_averages_by_day_cre(full_table, mouse_summary, version):
    sst_color = (158/255,218/255,229/255)
    vip_color = (197/255,176/255,213/255)
    slc_color = (255/255,152/255,150/255)

    sst_mice = mouse_summary.query('cre_line == "Sst-IRES-Cre"').copy()
    vip_mice = mouse_summary.query('cre_line == "Vip-IRES-Cre"').copy()
    slc_mice = mouse_summary.query('cre_line == "Slc17a7-IRES2-Cre"').copy()
    sst_mice_ids = sst_mice.index.values
    vip_mice_ids = vip_mice.index.values
    slc_mice_ids = slc_mice.index.values
    sst = full_table.query('donor_id in @sst_mice_ids').copy()
    vip = full_table.query('donor_id in @vip_mice_ids').copy()
    slc = full_table.query('donor_id in @slc_mice_ids').copy()

    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='strategy_dropout_index',color=sst_color,group_label='_cre',min_sessions=5)   
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='strategy_dropout_index',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='strategy_dropout_index',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='visual_only_dropout_index',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='visual_only_dropout_index',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='visual_only_dropout_index',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='timing_only_dropout_index',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='timing_only_dropout_index',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='timing_only_dropout_index',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='strategy_weight_index',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='strategy_weight_index',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='strategy_weight_index',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='avg_weight_task0',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='avg_weight_task0',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='avg_weight_task0',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='avg_weight_timing1D',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='avg_weight_timing1D',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='avg_weight_timing1D',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)   
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='avg_weight_bias',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='avg_weight_bias',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='avg_weight_bias',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)   
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='lick_hit_fraction',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='lick_hit_fraction',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='lick_hit_fraction',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='num_hits',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='num_hits',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='num_hits',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)   
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='session_roc',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='session_roc',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='session_roc',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(sst,sst_mice,version,label='Sst', metric='fraction_engaged',color=sst_color,group_label='_cre',min_sessions=5)    
    plot_average_by_day(vip,vip_mice,version,label='Vip', metric='fraction_engaged',fig=plt.gcf(),color=vip_color,group_label='_cre',min_sessions=5)
    plot_average_by_day(slc,slc_mice,version,label='Slc', metric='fraction_engaged',fig=plt.gcf(),color=slc_color,group_label='_cre',min_sessions=5)   

def plot_average_by_day(full_table,mouse_summary, version,min_sessions=20,group_label='',metric='strategy_dropout_index',method ='difference',fig=None,color='k',label=None):
    '''
        Makes a plot that computes sumary metrics of each mouse's strategy index across training days. 
        min_sessions is the minimum number of sessions for each day to compute the correlation
        metric = (difference, distance, abs_distance, correlation)
    '''    
    mouse_pivot = get_mouse_pivot_table(full_table, mouse_summary, metric=metric)


    # Build Plot
    if fig is None:
        plt.figure(figsize=(10,5))
        plt.axvspan(0,6,color='k',alpha=.1)
        plt.axhline(0, color='k',linestyle='--',alpha=0.5)
        plt.xlabel('Sessions before Ophys Stage 1',fontsize=16) 
        if metric in ['visual_only_dropout_index','avg_weight_timing1D']:
            plt.gca().invert_yaxis()

    # Iterate through training days
    first = True
    for dex,val in enumerate(full_table.pre_ophys_number.unique()): 
        if len(mouse_pivot[metric][val].unique())> min_sessions:
            if method == "difference":
                output = np.nanmean(mouse_pivot[metric][val])  
            elif method == "distance":
                output = np.nansum(np.sqrt(mouse_pivot['ophys_index']-mouse_pivot[metric][val]))/np.sum(~mouse_pivot[metric][val].isnull())
            elif method == "abs_distance":
                output = np.nansum(np.abs(mouse_pivot['ophys_index']-mouse_pivot[metric][val]))/np.sum(~mouse_pivot[metric][val].isnull())
            else:
                output = mouse_pivot['ophys_index'].corr(mouse_pivot[metric][val],method=method)
            if first & (label is not None):
                plt.plot(-val,output ,'o',color=color,label=label)
                first=False
                plt.legend()
            else:
                plt.plot(-val,output ,'o',color=color)



    plt.xlim(right=6)      
    # Clean up and save
    if  method in ['distance','abs_distance']:
        plt.ylabel(metric+' Distance ',fontsize=16)
    elif method =='difference':
        plt.ylabel(metric,fontsize=16)
    else:
        plt.ylabel(metric+' Correlation ',fontsize=16)

    directory = pgt.get_directory(version)
    plt.savefig(directory+'figures_training/avg_'+metric+'_by_day'+group_label+'.svg')
    plt.savefig(directory+'figures_training/avg_'+metric+'_by_day'+group_label+'.png')




