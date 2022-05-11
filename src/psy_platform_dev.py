import numpy as np
import psy_output_tools as po
import psy_analysis as pa
import matplotlib.pyplot as plt
import psy_style as pstyle
import psy_visualization as pv
import psy_metrics_tools as pm
import psy_general_tools as pgt
import psy_tools as ps
plt.ion()

BEHAVIOR_VERSION=20
EXAMPLE_BSID = 951520319
FIG_DIR = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/paper_figures/'

def make_engagement_figure():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)

    pa.RT_by_engagement(summary_df, BEHAVIOR_VERSION, title=None,density=True)
    plt.savefig(FIG_DIR+"RT_by_engagement.png")
    plt.savefig(FIG_DIR+"RT_by_engagement.svg")
    
    pv.plot_engagement_landscape(summary_df, BEHAVIOR_VERSION,savefig=True)
    plt.savefig(FIG_DIR+"engagement_landscape.png")
    plt.savefig(FIG_DIR+"engagement_landscape.svg")

    pv.plot_session_engagement(summary_df, EXAMPLE_BSID, BEHAVIOR_VERSION, savefig=True)
    plt.savefig(FIG_DIR+"engagement_example.png")
    plt.savefig(FIG_DIR+"engagement_example.svg")

def make_strategy_figure():
    summary_df = po.get_ophys_summary_table(BEHAVIOR_VERSION)

    pa.RT_by_group(summary_df, BEHAVIOR_VERSION, engaged=True,title=None)
    plt.savefig(FIG_DIR+"strategy_engagement.png")
    plt.savefig(FIG_DIR+"strategy_engagement.svg")

    pa.RT_by_group(summary_df, BEHAVIOR_VERSION, engaged=False,labels=['Visual Disengaged','Timing Disengaged'],title=None)
    plt.savefig(FIG_DIR+"strategy_disengagement.png")
    plt.savefig(FIG_DIR+"strategy_disengagement.svg")

    pv.plot_session_summary_weight_avg_scatter_task0(summary_df, version=BEHAVIOR_VERSION,savefig=True)    
    plt.savefig(FIG_DIR+"visual_post_omission_weight_scatter.png")
    plt.savefig(FIG_DIR+"visual_post_omission_weight_scatter.svg")

    pv.scatter_df(summary_df, 'dropout_task0','dropout_omissions1',version=BEHAVIOR_VERSION, plot_regression=True,flip1=True, flip2=True,plot_axis_lines=True)
    plt.savefig(FIG_DIR+"visual_post_omission_dropout_scatter.png")
    plt.savefig(FIG_DIR+"visual_post_omission_dropout_scatter.svg")
   
    pv.scatter_df(summary_df,'visual_only_dropout_index','timing_only_dropout_index',
        version=BEHAVIOR_VERSION,flip1=True,flip2=True, cindex='strategy_dropout_index') 
    plt.savefig(FIG_DIR+"visual_timing_dropout_scatter.png")
    plt.savefig(FIG_DIR+"visual_timing_dropout_scatter.svg")

