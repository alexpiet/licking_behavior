import psy_analysis as pa
import psy_general_tools as pgt
import psy_visualization as pv

BEHAVIOR_VERSION=20
EXAMPLE_BSID = 951520319
FIG_DIR = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/paper_figures/'

def make_engagement_figure():
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

def make_strategy_figure():
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

