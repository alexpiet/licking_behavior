import psy_general_tools as pgt
import psy_metrics_tools as pm
import psy_tools as ps
import psy_output_tools as po
import psy_analysis as pa

'''
psy_output_tools contains functions for generating the final output files
'''

# Define new version
po.build_id_fit_list(VERSION)

# Build summary tables 
po.build_summary_table(VERSION)
po.build_training_summary_table(VERSION)
po.build_mouse_summary_table(VERSION)

# Build summary files for each session, very slow
po.build_all_session_outputs(VERSION)
po.build_all_session_outputs(VERSION,TRAIN=True)

# Load summary tables
ophys_table = po.get_ophys_summary_table(version)
training_table = po.get_training_summary_table(version)
mouse_table = po.get_mouse_summary_table(version)

# Crash Analysis
crash_manifest = po.build_list_of_model_crashes(VERSION)
crash_manifest = po.build_list_of_train_model_crashes(VERSION)

