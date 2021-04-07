import psy_analyis as pa
import psy_output_tools as po

# Analyses that use the full summary table
ophys_table = po.get_ophys_summary_table(version)

# compare engaged/disengaged
pa.RT_by_engagement(ophys_table,version)
pa.RT_by_engagement(ophys_table.query('visual_strategy_session'),version,title='Visual')
pa.RT_by_engagement(ophys_table.query('not visual_strategy_session'),version,title='Timing')
pa.RT_by_engagement(ophys_table.query('visual_strategy_session'),version,title='Visual. Change', change_only=True)
pa.RT_by_engagement(ophys_table.query('not visual_strategy_session'),version,title='Timing. Change', change_only=True)

# compare groups of sessions or mice
pa.RT_by_group(ophys,title='all_images_strategy_engaged')
pa.RT_by_group(ophys,title='all_images_strategy_disengaged',engaged=False)
pa.RT_by_group(ophys,title='change_images_strategy_engaged',change_only=True)
pa.RT_by_group(ophys,title='change_images_strategy_disengaged',engaged=False,change_only=True)










