import psy_analyis as pa

# compare engaged/disengaged
pa.plot_all_RT_by_engagement(summary_df,version)

# compare groups of sessions or mice
pa.plot_all_RT_by_group(summary_df,version)

# Pivot plots, strategy relative to mouse average
pa.plot_all_pivoted(summary_df,version)








