import psy_general_tools as pgt
import psy_metrics_tools as pm
import psy_tools as ps
import psy_output_tools as po
from alex_utils import *

# Define model version, and output directory
output_dir = '/home/alex.piet/codebase/behavior/model_output/'
model_dir  = '/home/alex.piet/codebase/behavior/psy_fits_v9/'

# Get list of sessions
manifest = pgt.get_manifest()
ids = pgt.get_active_ids()

# Save results file for each session
po.build_all_session_outputs(ids,model_dir,output_dir):

# Build full summary table
po.build_summary_table(model_dir, output_dir)

#### Training data
train_manifest = pgt.get_training_manifest()
train_manifest = train_manifest.query('(not ophys) or (ophys & (stage == "0"))').copy()
train_ids = train_manifest.index.values

# Save results filee for each non ophys training session, including habituation
po.build_all_session_outputs(train_ids, model_dir, output_dir,TRAIN=True)


