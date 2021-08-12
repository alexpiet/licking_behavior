import numpy as np
import psy_output_tools as po
import psy_analysis as pa
import matplotlib.pyplot as plt
import psy_style as pstyle
import psy_metrics_tools as pm
import psy_general_tools as pgt
import psy_tools as ps
plt.ion()

BEHAVIOR_VERSION=20
EXAMPLE_SESSION = 0
FIG_DIR = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/platform_figures/'

def make_engagement_figure():
    ophys = po.get_ophys_summary_table(BEHAVIOR_VERSION)

    pa.RT_by_engagement(ophys, BEHAVIOR_VERSION, title=None,density=True)
    plt.savefig(FIG_DIR+"RT_by_engagement.png")
    plt.savefig(FIG_DIR+"RT_by_engagement.svg")
    
    pm.plot_engagement_landscape(ophys,plot_threshold=True)
    plt.savefig(FIG_DIR+"engagement_landscape.png")
    plt.savefig(FIG_DIR+"engagement_landscape.svg")

    pm.plot_metrics_from_table(ophys,EXAMPLE_SESSION)
    plt.savefig(FIG_DIR+"engagement_example.png")
    plt.savefig(FIG_DIR+"engagement_example.svg")

def make_strategy_figure():
    ophys = po.get_ophys_summary_table(BEHAVIOR_VERSION)
    ids = np.unique(ophys.behavior_session_id.values)
    pa.RT_by_group(ophys, BEHAVIOR_VERSION, engaged=True,title=None)
    plt.savefig(FIG_DIR+"strategy_engagement.png")
    plt.savefig(FIG_DIR+"strategy_engagement.svg")

    pa.RT_by_group(ophys, BEHAVIOR_VERSION, engaged=False,labels=['Visual Disengaged','Timing Disengaged'],title=None)
    plt.savefig(FIG_DIR+"strategy_disengagement.png")
    plt.savefig(FIG_DIR+"strategy_disengagement.svg")

    ps.plot_session_summary_weight_avg_scatter_task0(ids, version=BEHAVIOR_VERSION)    
    plt.savefig(FIG_DIR+"visual_post_omission_weight_scatter.png")
    plt.savefig(FIG_DIR+"visual_post_omission_weight_scatter.svg")

    ps.scatter_manifest(ophys, 'dropout_task0','dropout_omissions1',version=BEHAVIOR_VERSION, plot_regression=True,sflip1=True, sflip2=True,plot_axis_lines=True)
    plt.savefig(FIG_DIR+"visual_post_omission_dropout_scatter.png")
    plt.savefig(FIG_DIR+"visual_post_omission_dropout_scatter.svg")

    # generates a lot of figures. I need to organize some intermediate data folder
    #drop_dex, drop_var = ps.PCA_dropout(ids, pgt.get_mice_ids(), 20, ms=4)
    
    ps.plot_visual_vs_timing_dropout(ophys, BEHAVIOR_VERSION)   
    plt.savefig(FIG_DIR+"visual_timing_dropout_scatter.png")
    plt.savefig(FIG_DIR+"visual_timing_dropout_scatter.svg")


def make_strategy_supplement():
    # ROC
    # ROC comparison with static
    # priors
    # dropouts
    # avg. weights
    # weight trajectory
    # lick hit fraction
    return
