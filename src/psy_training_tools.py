import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import psy_tools as ps
import psy_style as pstyle
import psy_output_tools as po
import psy_general_tools as pgt
import matplotlib.pyplot as plt
from allensdk.brain_observatory.behavior.behavior_project_cache import \
    VisualBehaviorOphysProjectCache
plt.ion()

BEHAVIOR_VERSION = 21
TRAINING_VERSION = 22

def dev_notes():
    # Get a list of training sessions 
    train_manifest = ptt.get_training_manifest()

    # Get inventory of what sessions have been fit
    inventory = ptt.get_training_inventory()

    # Build a summary table with fit information
    training_summary = ptt.build_training_summary_table(TRAINING_VERSION)

    # Load the existing summary table
    training_summary = ptt.get_training_summary_table(TRAINING_VERSION)
 
    # outdated below here
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    full = ptt.get_full_behavior_table(training_summary, summary_df)
    
    # Plot Averages by training stage 
    ptt.plot_average_by_stage(full, metric='strategy_dropout_index')
    ptt.plot_average_by_stage(full, metric='strategy_dropout_index',
        plot_strategy=False)
    
    # Plot Average by Training session
    ptt.plot_average_by_day(full,metric='strategy_dropout_index')

def get_training_manifest(non_ophys=True,include_non_flashed=False):
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

    # Remove early non flashed sessions
    if not include_non_flashed:
        names = ['TRAINING_0_gratings_autorewards_15min','TRAINING_1_gratings']
        training['non_flashed'] = [x in names for x in training.session_type]
        training = training.query('not non_flashed').drop(columns=['non_flashed']).copy()

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
        version = TRAINING_VERSION

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
    training_summary = add_time_aligned_training_info(training_summary, version)

    print('Adding engagement information')
    training_summary = add_training_engagement_metrics(training_summary)
    training_summary = engagement_updates(training_summary)    

    print('Annotating mouse strategy')
    training_summary = add_mouse_strategy(training_summary)

    print('Saving')
    model_dir = pgt.get_directory(version,subdirectory='summary') 
    training_summary.to_pickle(model_dir+'_training_summary_table.pkl')

    return training_summary


def build_core_training_table(version):
    df = get_training_manifest()

    df['behavior_fit_available'] = df['ophys'] # copying size
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try:
            fit = ps.load_fit(row.behavior_session_id, version=version)
        except:
            df.at[index,'behavior_fit_available'] = False
        else:
            df.at[index,'behavior_fit_available'] = True
            df.at[index,'session_roc'] = ps.compute_model_roc(fit)
            df.at[index,'num_trial_false_alarm'] = \
                np.sum(fit['psydata']['full_df']['false_alarm'])
            df.at[index,'num_trial_correct_reject'] = \
                np.sum(fit['psydata']['full_df']['correct_reject'])

            # Get Strategy indices
            model_dex, taskdex,timingdex = ps.get_timing_index_fit(fit) 
            df.at[index,'strategy_dropout_index'] = model_dex
            df.at[index,'visual_only_dropout_index'] = taskdex
            df.at[index,'timing_only_dropout_index'] = timingdex

            # For each strategy add the hyperparameter, dropout score, and average weight
            dropout_dict_cv = ps.get_session_dropout(fit,cross_validation=True)
            dropout_dict_ev = ps.get_session_dropout(fit,cross_validation=False)
            sigma = fit['hyp']['sigma']
            wMode = fit['wMode']
            weights = ps.get_weights_list(fit['weights'])
            for dex, weight in enumerate(weights):
                df.at[index, 'prior_'+weight] =sigma[dex]
            for dex, weight in enumerate(weights):
                df.at[index, 'dropout_cv_'+weight] = dropout_dict_cv[weight]
                df.at[index, 'dropout_'+weight] = dropout_dict_ev[weight]
            for dex, weight in enumerate(weights):
                df.at[index, 'avg_weight_'+weight] = np.mean(wMode[dex,:])

    # Return only for sessions with fits
    print(str(len(df.query('not behavior_fit_available')))+\
        " sessions without model fits")
    df = df.query('behavior_fit_available').copy()
    
    # Compute classify session
    df['visual_strategy_session'] = -df['visual_only_dropout_index'] > \
        -df['timing_only_dropout_index']

    return df


def add_time_aligned_training_info(summary_df, version):
    
    # Initializing empty columns
    weight_columns = pgt.get_strategy_list(version)
    columns = {'hit','miss','image_false_alarm','image_correct_reject',
        'is_change', 'omitted', 'lick_bout_rate','reward_rate','RT','engaged',
        'lick_bout_start','image_index'} 
    for column in weight_columns:
        summary_df['weight_'+column] = [[]]*len(summary_df)
    for column in columns:
        summary_df[column] = [[]]*len(summary_df)      
    summary_df['strategy_weight_index_by_image'] = [[]]*len(summary_df)
    summary_df['lick_hit_fraction_rate'] = [[]]*len(summary_df)

    crash = 0
    for index, row in tqdm(summary_df.iterrows(),total=summary_df.shape[0]):
        try:
            strategy_dir = pgt.get_directory(version, subdirectory='strategy_df')
            session_df = pd.read_csv(strategy_dir+str(row.behavior_session_id)+'.csv')
        except Exception as e:
            crash +=1
            print(e)
            for column in weight_columns:
                summary_df.at[index, 'weight_'+column] = np.array([np.nan]*4800)
            for column in columns:
                summary_df.at[index, column] = np.array([np.nan]*4800) 
            summary_df.at[index, column] = np.array([np.nan]*4800)
        else: 
            # Add session level metrics
            summary_df.at[index,'num_hits'] = session_df['hit'].sum()
            summary_df.at[index,'num_miss'] = session_df['miss'].sum()
            summary_df.at[index,'num_omission_licks'] = \
                np.sum(session_df['omitted'] & session_df['lick_bout_start']) 
            summary_df.at[index,'num_post_omission_licks'] = \
                np.sum(session_df['omitted'].shift(1,fill_value=False) & \
                session_df['lick_bout_start'])
            summary_df.at[index,'num_late_task_licks'] = \
                np.sum(session_df['is_change'].shift(1,fill_value=False) & \
                session_df['lick_bout_start'])
            summary_df.at[index,'num_changes'] = session_df['is_change'].sum()
            summary_df.at[index,'num_omissions'] = session_df['omitted'].sum()
            summary_df.at[index,'num_image_false_alarm'] = \
                session_df['image_false_alarm'].sum()
            summary_df.at[index,'num_image_correct_reject'] = \
                session_df['image_correct_reject'].sum()
            summary_df.at[index,'num_lick_bouts'] = session_df['lick_bout_start'].sum()
            summary_df.at[index,'lick_fraction'] = session_df['lick_bout_start'].mean()
            summary_df.at[index,'omission_lick_fraction'] = \
                summary_df.at[index,'num_omission_licks']/\
                summary_df.at[index,'num_omissions'] 
            summary_df.at[index,'post_omission_lick_fraction'] = \
                summary_df.at[index,'num_post_omission_licks']/\
                summary_df.at[index,'num_omissions'] 
            summary_df.at[index,'lick_hit_fraction'] = \
                session_df['rewarded'].sum()/session_df['lick_bout_start'].sum() 
            summary_df.at[index,'trial_hit_fraction'] = \
                session_df['rewarded'].sum()/session_df['is_change'].sum() 

            # Add time aligned information
            for column in weight_columns:
                summary_df.at[index, 'weight_'+column] = \
                    pgt.get_clean_rate(session_df[column].values)
            for column in columns:
                summary_df.at[index, column] = \
                    pgt.get_clean_rate(session_df[column].values)
            summary_df.at[index,'lick_hit_fraction_rate'] = \
                pgt.get_clean_rate(session_df['lick_hit_fraction'].values)

            # Compute strategy indexes
            summary_df.at[index,'strategy_weight_index_by_image'] = \
                pgt.get_clean_rate(session_df['task0'].values) - \
                pgt.get_clean_rate(session_df['timing1D'].values) 
            summary_df.at[index,'strategy_weight_index'] = \
                np.nanmean(summary_df.at[index,'strategy_weight_index_by_image'])

    if crash > 0:
        print(str(crash) + ' sessions crashed')
    return summary_df 


def add_training_engagement_metrics(summary_df,min_engaged_fraction=0.05):

    # Add Engaged specific metrics
    summary_df['fraction_engaged'] = \
        [np.nanmean(summary_df.loc[x]['engaged']) for x in summary_df.index.values]

    # Add average value of strategy weights split by engagement stats
    columns = {
        'task0':'visual',
        'timing1D':'timing',
        'bias':'bias'}
    for k in columns.keys():  
        summary_df[columns[k]+'_weight_index_engaged'] = \
        [np.nanmean(summary_df.loc[x]['weight_'+k][summary_df.loc[x]['engaged'] == True]) 
            if summary_df.loc[x]['fraction_engaged'] > min_engaged_fraction else np.nan 
            for x in summary_df.index.values]
        summary_df[columns[k]+'_weight_index_disengaged'] = \
        [np.nanmean(summary_df.loc[x]['weight_'+k][summary_df.loc[x]['engaged'] == False])
            if summary_df.loc[x]['fraction_engaged'] < 1-min_engaged_fraction else np.nan 
            for x in summary_df.index.values]
    summary_df['strategy_weight_index_engaged'] = \
        summary_df['visual_weight_index_engaged'] -\
        summary_df['timing_weight_index_engaged']
    summary_df['strategy_weight_index_disengaged'] = \
        summary_df['visual_weight_index_disengaged'] -\
        summary_df['timing_weight_index_disengaged']

    # Add average value of columns split by engagement state
    columns = {'lick_bout_rate','reward_rate','lick_hit_fraction_rate','hit',
        'miss','image_false_alarm','image_correct_reject','RT'}
    for column in columns: 
        summary_df[column+'_engaged'] = \
        [np.nanmean(summary_df.loc[x][column][summary_df.loc[x]['engaged'] == True]) 
            if summary_df.loc[x]['fraction_engaged'] > min_engaged_fraction else np.nan
            for x in summary_df.index.values]
        summary_df[column+'_disengaged'] = \
        [np.nanmean(summary_df.loc[x][column][summary_df.loc[x]['engaged'] == False]) 
            if (summary_df.loc[x]['fraction_engaged'] < 1-min_engaged_fraction) &
            (not np.all(np.isnan(summary_df.loc[x][column][summary_df.loc[x]['engaged']==False]))) 
            else np.nan for x in summary_df.index.values]
    return summary_df


def engagement_updates(summary_df):
    summary_df['engagement_v1'] = [[]]*len(summary_df)      
    summary_df['engagement_v2'] = [[]]*len(summary_df)      
    summary_df['engagement_v1'] = summary_df['engaged']
    v2 = []
    for index, row in tqdm(summary_df.iterrows(), total=summary_df.shape[1]):
        this = np.array([x[0] and x[1] > 0.1 \
            for x in zip(row.engagement_v1, row.lick_bout_rate)])
        v2.append(this)
    summary_df['engagement_v2'] = v2
    return summary_df


def add_mouse_strategy(df):
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)   
    mouse = summary_df.query('experience_level == "Familiar"')\
        .groupby('mouse_id')['strategy_dropout_index'].mean().to_frame()
    mouse['visual_mouse'] = mouse['strategy_dropout_index'] > 0
    df = pd.merge(df, 
        mouse[['visual_mouse']], on=['mouse_id'])
    return df


def get_full_behavior_table(train_summary, ophys_summary):

    # Add mouse strategy and pre-ophys counts to ophys df
    ophys_summary = ophys_summary.copy()
    ophys_summary = add_mouse_strategy(ophys_summary)
    ophys_summary['pre_ophys_number'] = -ophys_summary\
        .groupby(['mouse_id']).cumcount(ascending=True)

    # Merge tables
    train_summary['experience_level'] = train_summary['session_type']
    full_table = train_summary.query('pre_ophys_number > 0').copy()
    full_table = full_table.append(ophys_summary,sort=False)
    full_table = full_table.sort_values(by='behavior_session_id')\
        .reset_index(drop=True)

    return full_table 



## Building stimulus tables for non-flashed stimuli 
################################################################################

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


## Plotting Functions
################################################################################

def plot_average_by_stage_inner(group,color='k',label=None,skip=[],alpha=.2):
    group['std_err'] = group['std']/np.sqrt(group['count'])
    for index, row in group.iterrows():
        if (index not in skip) & (index[1:] not in skip):
            if index in ['TRAINING_2 ','TRAINING_3 ','TRAINING_4h', 
                'TRAINING_5h','_OPHYS_1','_OPHYS_3','_OPHYS_4',
                '_OPHYS_6','_OPHYS_0','Familiar','Novel 1','Novel +']:
                plt.plot(row['mean'],row.order,'o',zorder=3,color=color)
            else:       
                plt.plot(row['mean'],row.order,'o',color=color,alpha=alpha,zorder=3)
            plt.plot([row['mean']-row['std_err'], row['mean']+row['std_err']],
                [row.order, row.order], '-',alpha=alpha,zorder=2,color=color)
            if index == 'TRAINING_2 ':
                plt.plot(row['mean'],row.order,'o',zorder=3,color=color,label=label)

def plot_average_by_stage(full_table,metric='strategy_dropout_index',
    savefig=False,version=None,flip_axis = False,filetype='.svg',
    plot_strategy=True,plot_cre=False,skip=[],alpha=.2,metric_name=''):
    
    full_table['clean_session_type'] = [
        clean_session_type(x) for x in full_table.experience_level]
    session_order = {
        'TRAINING_2 ':0,
        'TRAINING_3 ':1,
        'TRAINING_4 ':2,
        'TRAINING_4h':3,
        'TRAINING_4l':4,
        'TRAINING_5 ':5,
        'TRAINING_5h':6,
        'TRAINING_5l':7,
        '_OPHYS_0':8,
        'Familiar':9,
        'Novel 1':10,
        'Novel +':11
        }
    colors = pstyle.get_colors()

    plt.figure(figsize=(6.5,3.75))
    if (not plot_strategy) & (not plot_cre):
        # Plot average across all groups
        group = full_table.groupby('clean_session_type')[metric].describe()
        group['order'] = [session_order[x] for x in group.index.values]
        group=group.sort_values(by='order')
        plot_average_by_stage_inner(group,skip=skip,alpha=alpha)

    elif plot_strategy:
        # Plot Visual Mice
        visual_color = colors['visual']
        visual = full_table.query('visual_mouse').copy()
        group = visual.groupby('clean_session_type')[metric].describe()
        group['order'] = [session_order[x] for x in group.index.values]
        group=group.sort_values(by='order')
        plot_average_by_stage_inner(group,color=visual_color,
            label='Visual Ophys Mice',skip=skip,alpha=alpha)

        # Plot Timing Mice
        timing_color = colors['timing'] 
        timing = full_table.query('not visual_mouse').copy()
        group = timing.groupby('clean_session_type')[metric].describe()
        group['order'] = [session_order[x] for x in group.index.values]
        group=group.sort_values(by='order')
        plot_average_by_stage_inner(group,color=timing_color,
            label='Timing Ophys Mice',skip=skip,alpha=alpha)
    else:
        # plot cre lines
        sst_color = colors['Sst-IRES-Cre'] 
        vip_color = colors['Vip-IRES-Cre'] 
        slc_color = colors['Slc17a7-IRES2-Cre'] 
        sst = full_table.query('cre_line == "Sst-IRES-Cre"').copy()
        vip = full_table.query('cre_line == "Vip-IRES-Cre"').copy()
        slc = full_table.query('cre_line == "Slc17a7-IRES2-Cre"').copy()
        group = vip.groupby('clean_session_type')[metric].describe()
        group['order'] = [session_order[x] for x in group.index.values]
        group=group.sort_values(by='order')
        plot_average_by_stage_inner(group,color=vip_color,label='Vip',
            skip=skip,alpha=alpha)
        group = sst.groupby('clean_session_type')[metric].describe()
        group['order'] = [session_order[x] for x in group.index.values]
        group=group.sort_values(by='order')
        plot_average_by_stage_inner(group,color=sst_color,label='Sst',
            skip=skip,alpha=alpha)
        group = slc.groupby('clean_session_type')[metric].describe()
        group['order'] = [session_order[x] for x in group.index.values]
        group=group.sort_values(by='order')
        plot_average_by_stage_inner(group,color=slc_color,label='Slc',
            skip=skip,alpha=alpha)

    # Clean up plot
    if flip_axis:
        plt.gca().invert_xaxis()

    labels = [x[1:] if x[0] == "_" else x for x in group.index.values]
    labels = [x for x in labels if x not in skip]
    style = pstyle.get_style()
    plt.gca().set_yticks(np.arange(0,len(labels)))
    plt.gca().set_yticklabels(labels,rotation=0,fontsize=12)   
    plt.axvline(0,color=style['axline_color'],
        alpha=style['axline_alpha'],
        ls=style['axline_linestyle'])
    plt.axhline(8.5,color=style['axline_color'],
        alpha=style['axline_alpha'],
        ls=style['axline_linestyle'])
    plt.xlabel(pgt.get_clean_string([metric])[0],fontsize=16)
    plt.gca().xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    plt.gca().yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    
    if plot_strategy or plot_cre:
        plt.legend()
    if metric =='session_roc':
        plt.xlim([.6,1])

    plt.tight_layout()
    if savefig:
        directory = pgt.get_directory(version)
        if plot_strategy:
            plt.savefig(directory+'figures_training/mouse_groups_'+metric+\
                '_by_stage'+filetype)
        elif plot_cre:
            plt.savefig(directory+'figures_training/cre_'+metric+'_by_stage'+filetype)
        else:
            plt.savefig(directory+'figures_training/avg_'+metric+'_by_stage'+filetype)

def clean_session_type(session_type):
    #raise Exception('Need to update')
    sessions = {
    "Familiar":                          "Familiar",
    "Novel 1":                           "Novel 1",
    "Novel >1":                          "Novel +",
    "OPHYS_0_images_A_habituation":      "_OPHYS_0",
    "OPHYS_0_images_B_habituation":      "_OPHYS_0",
    "OPHYS_1_images_A":                  "_OPHYS_1",
    "OPHYS_1_images_B":                  "_OPHYS_1",
    "OPHYS_3_images_A":                  "_OPHYS_3",
    "OPHYS_3_images_B":                  "_OPHYS_3",
    "OPHYS_4_images_A":                  "_OPHYS_4",
    "OPHYS_4_images_B":                  "_OPHYS_4",
    "OPHYS_6_images_A":                  "_OPHYS_6",
    "OPHYS_6_images_B":                  "_OPHYS_6",
    "TRAINING_0_gratings_autorewards_15min":"TRAINING_0",
    "TRAINING_1_gratings":               "TRAINING_1 ",
    "TRAINING_2_gratings_flashed":       "TRAINING_2 ",
    "TRAINING_3_images_A_10uL_reward":   "TRAINING_3 ",
    "TRAINING_3_images_B_10uL_reward":   "TRAINING_3 ",
    "TRAINING_4_images_A_handoff_lapsed":"TRAINING_4l",
    "TRAINING_4_images_B_handoff_lapsed":"TRAINING_4l",
    "TRAINING_4_images_A_handoff_ready": "TRAINING_4h",
    "TRAINING_4_images_B_handoff_ready": "TRAINING_4h",
    "TRAINING_4_images_A_training":      "TRAINING_4 ",
    "TRAINING_4_images_B_training":      "TRAINING_4 ",
    "TRAINING_5_images_A_handoff_lapsed":"TRAINING_5l",
    "TRAINING_5_images_B_handoff_lapsed":"TRAINING_5l",
    "TRAINING_5_images_A_handoff_ready": "TRAINING_5h",
    "TRAINING_5_images_B_handoff_ready": "TRAINING_5h",
    "TRAINING_5_images_A_training":      "TRAINING_5 ",
    "TRAINING_5_images_B_training":      "TRAINING_5 ",
    "TRAINING_5_images_A_epilogue":      "TRAINING_5 ",
    "TRAINING_5_images_B_epilogue":      "TRAINING_5 "
    }
    return sessions[session_type]


def plot_average_by_day_inner(ax, df,metric,numbering, min_sessions,color):
    g = df.groupby(numbering)[metric].mean().to_frame()
    g['count'] = df.groupby(numbering)[metric].count()
    g['sem'] = df.groupby(numbering)[metric].std()
    g['sem'] = g['sem']/np.sqrt(g['count'])
    g=g[g['count']>min_sessions]
    ax.plot(g.index.values,g[metric],'o',color=color)
    ax.errorbar(g.index.values,g[metric],g['sem'],fmt='none',color=color,alpha=.5)

def plot_average_by_day(full_table,metric='strategy_dropout_index',
    split='visual_mouse',min_sessions=5,numbering='pre_ophys_number',savefig=False):
    '''
        Plot average metric by training day
    '''   

    full_table = full_table.copy()
    full_table['session_num'] = -full_table['pre_ophys_number']
    full_table.at[full_table['experience_level']=="Familiar",'session_num']=0
    full_table.at[full_table['experience_level']=="Novel 1",'session_num']=1
    full_table.at[full_table['experience_level']=="Novel >1",'session_num']=2
    numbering = 'session_num'

    # Build Plot
    fig, ax = plt.subplots(figsize=(10,5))
    plt.axhline(0, color='k',linestyle='--',alpha=0.5)
    plt.axvline(-.5, color='k',linestyle='--',alpha=0.5)
    plt.xlabel('sessions before imaging',fontsize=16) 
    if metric in ['visual_only_dropout_index','avg_weight_timing1D']:
        ax.invert_yaxis()
    colors = pstyle.get_colors()

    # Iterate through training days
    if split == 'visual_mouse':
        plot_average_by_day_inner(ax, full_table.query('visual_mouse'),metric, numbering,
            min_sessions,colors['visual'])
        plot_average_by_day_inner(ax, full_table.query('not visual_mouse'),metric, 
            numbering,min_sessions,colors['timing'])
    
    # Clean up and save
    plt.ylabel(pgt.get_clean_string([metric])[0],fontsize=16)
    plt.xlim(-40,3)
    style =pstyle.get_style()
    ax.xaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.yaxis.set_tick_params(labelsize=style['axis_ticks_fontsize'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xticks([-40,-35,-30,-25,-20,-15,-10,-5,0,1,2])
    labels = ['-40','-35','-30','-25','-20','-15','-10','-5','F','N','+']
    ax.set_xticklabels(labels,rotation=0,fontsize=12)   


    if savefig:
        directory = pgt.get_directory(version)
        plt.savefig(directory+'figures_training/avg_'+metric+'_by_day'+group_label+'.svg')
        plt.savefig(directory+'figures_training/avg_'+metric+'_by_day'+group_label+'.png')




