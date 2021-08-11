import psy_output_tools as po
import psy_analysis as pa
import matplotlib.pyplot as plt
import psy_style as pstyle
plt.ion()

BEHAVIOR_VERSION=20
FIG_DIR = '/allen/programs/braintv/workgroups/nc-ophys/alex.piet/behavior/platform_figures/'

def make_engagement_figure():
    ophys = po.get_ophys_summary_table(BEHAVIOR_VERSION)

    pa.RT_by_engagement(ophys, BEHAVIOR_VERSION, title=None,density=True)
    plt.savefig(FIG_DIR+"RT_by_engagement.png")
    plt.savefig(FIG_DIR+"RT_by_engagement.svg")


