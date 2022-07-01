import psy_analysis as pa
import psy_general_tools as pgt
import psy_visualization as pv
import psy_output_tools as po
import matplotlib.pyplot as plt
import build_timing_regressor as b

BEHAVIOR_VERSION=21
EXAMPLE_BSID = 951520319
FIG_DIR = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/paper_figures/'

'''
    Figure 1 Supplement - basic behavior
        num hits, num false alarms?
        licking rate etc over session

    Figure 2 Supplement - Model validation
        ROC scatter against static ROC
        ROC scatter against strategy index
        strategy index against lick-hit-fraction
        
    Figure 2 Supplement - PCA
        Show pca components
        show VE by PC #
        show VE by strategy index
        
    Figure 2 Supplement - Strategy correlations
        show each strategy correlated with num hits, etc

    Figure 2 Supplement - Novelty
        Show three plots about average behavior
        show trajectory over session
'''

def make_figure_1_timing_regressor():
    b.build_timing_schematic(version=BEHAVIOR_VERSION, savefig=True)
    df = b.build_timing_regressor(version=BEHAVIOR_VERSION, savefig=True)

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

def make_figure_2_supplement_pca():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    pa.compute_PCA(summary_df, version=BEHAVIOR_VERSION,on='dropout',
        savefig=True)

def make_figure_3():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    pv.plot_engagement_landscape(summary_df,version,savefig=True, filetype='.png')
    pv.RT_by_engagement(summary_df,BEHAVIOR_VERSION,savefig=True, filetype='.svg')
    pv.RT_by_group(summary_df,BEHAVIOR_VERSION,engaged='engaged',ylim=.0031,
        savefig=True, filetype='.svg')
    pv.RT_by_group(summary_df,BEHAVIOR_VERSION,engaged='disengaged',ylim=.0031,
        savefig=True, filetype='.svg')
    pv.plot_session_summary_trajectory(summary_df,'engaged',version=BEHAVIOR_VERSION,
        categories='visual_strategy_session',savefig=True, filetype='.svg')

def dev_make_engagement_figure():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)

    pv.RT_by_engagement(summary_df, BEHAVIOR_VERSION, density=True,savefig=True)
    plt.savefig(FIG_DIR+"RT_by_engagement.png")
    plt.savefig(FIG_DIR+"RT_by_engagement.svg")
    
    pv.plot_engagement_landscape(summary_df, BEHAVIOR_VERSION,savefig=True)
    plt.savefig(FIG_DIR+"engagement_landscape.png")
    plt.savefig(FIG_DIR+"engagement_landscape.svg")

    session = pgt.get_data(EXAMPLE_BSID)
    pv.plot_session_engagement(session, BEHAVIOR_VERSION, savefig=True)
    plt.savefig(FIG_DIR+"engagement_example.png")
    plt.savefig(FIG_DIR+"engagement_example.svg")

def dev_make_strategy_figure():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)

    pv.RT_by_group(summary_df, BEHAVIOR_VERSION, engaged=True)
    plt.savefig(FIG_DIR+"strategy_engagement.png")
    plt.savefig(FIG_DIR+"strategy_engagement.svg")

    pv.RT_by_group(summary_df, BEHAVIOR_VERSION, engaged=False)
    plt.savefig(FIG_DIR+"strategy_disengagement.png")
    plt.savefig(FIG_DIR+"strategy_disengagement.svg")

    pv.plot_session_summary_weight_avg_scatter_task0(summary_df, 
        version=BEHAVIOR_VERSION,savefig=True)    
    plt.savefig(FIG_DIR+"visual_post_omission_weight_scatter.png")
    plt.savefig(FIG_DIR+"visual_post_omission_weight_scatter.svg")

    pv.scatter_df(summary_df, 'dropout_task0','dropout_omissions1',
        version=BEHAVIOR_VERSION, plot_regression=True,flip1=True, 
        flip2=True,plot_axis_lines=True,savefig=True)
    plt.savefig(FIG_DIR+"visual_post_omission_dropout_scatter.png")
    plt.savefig(FIG_DIR+"visual_post_omission_dropout_scatter.svg")
   
    pv.scatter_df(summary_df,'visual_only_dropout_index','timing_only_dropout_index',
        version=BEHAVIOR_VERSION,flip1=True,flip2=True, cindex='strategy_dropout_index',
        savefig=True) 
    plt.savefig(FIG_DIR+"visual_timing_dropout_scatter.png")
    plt.savefig(FIG_DIR+"visual_timing_dropout_scatter.svg")

