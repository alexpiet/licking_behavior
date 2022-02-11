import psy_analyis as pa
import psy_output_tools as po

# Analyses that use the full summary table
ophys = po.get_ophys_summary_table(version)

# compare engaged/disengaged
pa.plot_all_RT_by_engagement(ophys,version)

# compare groups of sessions or mice
pa.plot_all_RT_by_group(ophys,version)

# Pivot plots, strategy relative to mouse average
pa.plot_all_pivoted(model_manifest,version)








