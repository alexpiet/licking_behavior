import psy_output_tools as po
import psy_analysis as pa
import matplotlib.pyplot as plt
import psy_style as pstyle
import psy_metrics_tools as pm
import psy_general_tools as pgt
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

