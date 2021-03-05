import psy_general_tools as pgt
import psy_metrics_tools as pm
import psy_tools as ps
import psy_output_tools as po

# Define new version
VERSION = 20
po.build_id_fit_list(VERSION)

# Ophys outputs
po.build_summary_table(VERSION)
po.build_all_session_outputs(VERSION)

# training outputs
po.build_training_summary_table(VERSION)
po.build_all_session_outputs(VERSION,TRAIN=True)

# Load summary tables
model_manifest          = pd.read_csv(po.OUTPUT_DIR+'_summary_table.csv')
training_model_manifest = pd.read_csv(po.OUTPUT_DIR+'_training_summary_table.csv')

# Crash Analysis
crash_manifest = po.build_list_of_model_crashes(VERSION)
crash_manifest = po.build_list_of_train_model_crashes(VERSION)

