import psy_tools as ps
import psy_analysis as pa
import psy_general_tools as pgt
import psy_visualization as pv
import psy_output_tools as po
import matplotlib.pyplot as plt
import build_timing_regressor as b
import psy_metrics_tools as pm
import numpy as np

BEHAVIOR_VERSION=21
EXAMPLE_BSID = 951520319
FIG_DIR = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/paper_figures/'

def add_fit_prediction(session,version):
    # Annotate licks and bouts if not already done
    if 'bout_number' not in session.licks:
        pm.annotate_licks(session)
    if 'bout_start' not in session.stimulus_presentations:
        pm.annotate_bouts(session)
    if 'reward_rate' not in session.stimulus_presentations:
        pm.annotate_image_rolling_metrics(session)
    # Get full model prediction
    fit = ps.load_fit(session.metadata['behavior_session_id'], version)
    session.stimulus_presentations.at[\
        ~session.stimulus_presentations['in_lick_bout'], 'prediction'] = fit['ypred']
    win_dur=120
    win_type='boxcar'#'gaussian'
    win_std=30
    session.stimulus_presentations['prediction'] = \
        session.stimulus_presentations['prediction'].\
        rolling(win_dur,min_periods=1,win_type=win_type,center=True).\
        mean(std=win_std)
    session.stimulus_presentations['target'] = \
        session.stimulus_presentations['bout_start']
    session.stimulus_presentations.at[session.stimulus_presentations['in_lick_bout'],'target'] = np.nan
    session.stimulus_presentations['target'] = \
        session.stimulus_presentations['bout_start'].\
        rolling(win_dur,min_periods=1,win_type=win_type,center=True).\
        mean(std=win_std)

def make_figure_1_examples():
    session = pgt.get_data(EXAMPLE_BSID)
    pv.plot_strategy_examples(session, version=BEHAVIOR_VERSION, savefig=True)

def make_figure_1_diagram():
    #session = pgt.get_data(795742990)
    session = pgt.get_data(792680306)
    add_fit_prediction(session,BEHAVIOR_VERSION)
    pv.plot_session_metrics(session, plot_list=['target','prediction'],plot_example=True,
        version=BEHAVIOR_VERSION)
    fit = ps.load_fit(session.metadata['behavior_session_id'],BEHAVIOR_VERSION)
    pv.plot_session(session,detailed=True, fit=fit,x=[565],mean_center_strategies=False)

def make_figure_1_supplement_behavior():
    '''
        Figure 1 Supplement - basic behavior
            num hits, num false alarms?
            licking rate etc over session
    '''
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    event = ['reward_rate','RT']
    for e in event:
        pv.plot_session_summary_trajectory(summary_df,e,version=BEHAVIOR_VERSION,
            savefig=True, filetype='.svg')
    event = ['miss', 'image_false_alarm','image_correct_reject','hit']
    pv.plot_session_summary_multiple_trajectory(summary_df,event,
        version=BEHAVIOR_VERSION, savefig=True,filetype='.svg',event_names='responses')
    pv.histogram_df(summary_df,'num_hits',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'num_miss',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'num_omission_licks',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'num_post_omission_licks',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'num_image_false_alarm',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'num_lick_bouts',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'lick_fraction',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'omission_lick_fraction',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'post_omission_lick_fraction',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'trial_hit_fraction',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.histogram_df(summary_df,'lick_hit_fraction',version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')


def make_figure_1_timing_regressor():
    bsid = pgt.get_debugging_id(1)
    session = pgt.get_data(session)
    #b.build_timing_schematic(session, version=BEHAVIOR_VERSION, savefig=True)
    pv.plot_segmentation_schematic(session, savefig=True, version=BEHAVIOR_VERSION)
    df = b.build_timing_regressor(version=BEHAVIOR_VERSION, savefig=True)
    # TODO Issue #196
    # consider adding - The point being that timing is aligned to end of licking period
    #pv.plot_interlick_interval(bouts_df,key='pre_ibi',version=version,
    #    categories='post_reward')
    #pv.plot_interlick_interval(bouts_df,key='pre_ibi_from_start',version=version,
    #    categories='post_reward')
    #pv.plot_chronometric(bouts_df, version)


def make_figure_1_supplement_task():
    change_df = po.get_change_table(BEHAVIOR_VERSION)
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)

    pv.plot_image_pair_repetitions(change_df, BEHAVIOR_VERSION,savefig=True,
        filetype='.svg')
    pv.histogram_df(summary_df, 'num_changes',version=BEHAVIOR_VERSION,
        savefig=True,filetype='.svg')
    pv.plot_image_repeats(change_df, BEHAVIOR_VERSION,savefig=True,filetype='.svg')
    pv.histogram_df(summary_df, 'num_hits',version=BEHAVIOR_VERSION,
        savefig=True,filetype='.svg')


def make_figure_1_supplement_licking():
    licks_df = po.get_licks_table(BEHAVIOR_VERSION)
    bouts_df = po.build_bout_table(licks_df)
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    summary_df['all'] = True

    pv.plot_interlick_interval(licks_df, version=BEHAVIOR_VERSION,
        savefig=True,filetype='.svg')
    pv.plot_bout_durations(bouts_df, BEHAVIOR_VERSION, savefig=True,filetype='.svg')
    pv.RT_by_group(summary_df, BEHAVIOR_VERSION, groups=['all'], labels=[''],
        engaged=None, savefig=True, filetype='.svg')
    # TODO #196 consider adding
    #pv.plot_interlick_interval(bouts_df,key='pre_ibi',version=version,
    #categories='bout_rewarded')
    #pv.histogram_df(summary_df,'num_lick_bouts',version=version)


def make_figure_2():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    pv.plot_static_comparison(summary_df,version=BEHAVIOR_VERSION, savefig=True,
        filetype='.svg')
    pv.plot_session_summary_dropout(summary_df,version=BEHAVIOR_VERSION,savefig=True,
        cross_validation=False, filetype='.svg')
    pv.plot_session_summary_weights(summary_df,version=BEHAVIOR_VERSION,savefig=True,
        filetype='.svg')
    pv.plot_session_summary_priors(summary_df,version=BEHAVIOR_VERSION,savefig=True,
        filetype='.svg')
    pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index', 
        version=BEHAVIOR_VERSION,flip1=True,flip2=True,cindex='strategy_dropout_index',
        savefig=True,filetype='.svg')
    pv.plot_session_summary_weight_avg_scatter_task0(summary_df,version=BEHAVIOR_VERSION,
        savefig=True, filetype='.svg')
    pv.scatter_df_by_mouse(summary_df,'strategy_dropout_index',version=BEHAVIOR_VERSION,
        savefig=True,filetype='.svg')
    pv.scatter_df(summary_df, 'strategy_dropout_index','num_hits',
        cindex='strategy_dropout_index',version=BEHAVIOR_VERSION, 
        savefig=True, filetype='.svg')


def make_figure_2_supplement_model_validation():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    pv.plot_static_comparison(summary_df,version=BEHAVIOR_VERSION, savefig=True,
        filetype='.svg')
    pv.scatter_df(summary_df,'strategy_dropout_index','lick_hit_fraction', 
        version=BEHAVIOR_VERSION,savefig=True,filetype='.svg',figsize=(5,4))
    pv.scatter_df(summary_df,'strategy_dropout_index','session_roc', 
        version=BEHAVIOR_VERSION,savefig=True,filetype='.svg',figsize=(5,4))
    pv.scatter_df(summary_df,'visual_only_dropout_index','lick_hit_fraction', 
        version=BEHAVIOR_VERSION,savefig=True,filetype='.svg',figsize=(5,4))


def make_figure_2_supplement_strategy_characterization():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)   
    # Plot session-wise metrics against strategy weights
    event=['hits','miss','lick_fraction']
    for e in event:
        pv.plot_session_summary_weight_avg_scatter_task_events(summary_df,e,
        version=BEHAVIOR_VERSION,savefig=True,filetype='.svg')


def make_figure_2_supplement_strategy_characterization_rates():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)   
    # Plot image-wise metrics, averaged across sessions
    events = ['bias','task0','omissions','omissions1','timing1D']
    pv.plot_session_summary_multiple_trajectory(summary_df,events,
        version=BEHAVIOR_VERSION, savefig=True,filetype='.svg',event_names='strategies')
    pv.plot_session_summary_multiple_trajectory(\
        summary_df.query('visual_strategy_session'),events,
        version=BEHAVIOR_VERSION, savefig=True,filetype='.svg',
        event_names='strategies_visual')
    pv.plot_session_summary_multiple_trajectory(\
        summary_df.query('not visual_strategy_session'),events,
        version=BEHAVIOR_VERSION, savefig=True,filetype='.svg',
        event_names='strategies_timing')

    events = ['hit','miss']
    pv.plot_session_summary_multiple_trajectory(summary_df,events,
        version=BEHAVIOR_VERSION, savefig=True,filetype='.svg',event_names='task_events')
    for key in events:
        pv.plot_session_summary_trajectory(summary_df,key,BEHAVIOR_VERSION,
            categories='visual_strategy_session',savefig=True, filetype='.svg')
    
    events = ['lick_hit_fraction_rate','lick_bout_rate']
    pv.plot_session_summary_multiple_trajectory(summary_df,events,
        version=BEHAVIOR_VERSION, savefig=True,filetype='.svg',event_names='metrics')
    for key in events:
        pv.plot_session_summary_trajectory(summary_df,key,BEHAVIOR_VERSION,
            categories='visual_strategy_session',savefig=True, filetype='.svg')


def make_figure_2_supplement_pca():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    pa.compute_PCA(summary_df, version=BEHAVIOR_VERSION,on='dropout',
        savefig=True)


def make_figure_2_novelty():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)  
    pv.scatter_df_by_experience(summary_df, ['Familiar','Novel 1'],
        'strategy_dropout_index',experience_type='experience_level',
        version=BEHAVIOR_VERSION,savefig=True, filetype='.svg') 
    pv.histogram_df_by_experience(summary_df,['Familiar','Novel 1'],
        'strategy_dropout_index',experience_type='experience_level',
        version=BEHAVIOR_VERSION,savefig=True,filetype='.svg')
    pv.plot_pivoted_df_by_experience(summary_df,'strategy_dropout_index',
        version=BEHAVIOR_VERSION, savefig=True, filetype='.svg')
    keys = ['lick_hit_fraction_rate','task0','timing1D','lick_bout_rate']
    for key in keys:
        pv.plot_session_summary_trajectory(summary_df,key,BEHAVIOR_VERSION,
            categories='experience_level',savefig=True,filetype='.svg')


def make_figure_3():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    pv.plot_engagement_landscape(summary_df,version,savefig=True, filetype='.png')
    pv.plot_engagement_analysis(summary_df,version,savefig=True, filetype='.svg')
    pv.RT_by_engagement(summary_df,BEHAVIOR_VERSION,savefig=True, filetype='.svg')
    pv.RT_by_group(summary_df,BEHAVIOR_VERSION,engaged='engaged',ylim=.0031,
        savefig=True, filetype='.svg')
    pv.RT_by_group(summary_df,BEHAVIOR_VERSION,engaged='disengaged',ylim=.0031,
        savefig=True, filetype='.svg')
    pv.plot_session_summary_trajectory(summary_df,'engaged',version=BEHAVIOR_VERSION,
        categories='visual_strategy_session',savefig=True, filetype='.svg')


def make_figure_4_supplement_strategy_matched():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    pv.histogram_df(summary_df, 'strategy_dropout_index',categories='cre_line',
        savefig=True, version=BEHAVIOR_VERSION,filetype='.svg')
    pv.histogram_df(summary_df.query('strategy_matched'), 'strategy_dropout_index',
        categories='cre_line',savefig=True, version=BEHAVIOR_VERSION,
        filetype='.svg',group='strategy_matched')
    pv.scatter_df(summary_df, 'visual_only_dropout_index','timing_only_dropout_index',
        flip1=True, flip2=True,categories='cre_line',savefig=True,  
        version=BEHAVIOR_VERSION,filetype='.svg',figsize=(5,4))


