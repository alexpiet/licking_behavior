import psy_general_tools as pgt
import psy_metrics_tools as pm
import psy_tools as ps
import psy_output_tools as po
from alex_utils import *

# Define model version, and output directory
output_dir  = '/home/alex.piet/codebase/behavior/model_output/'
model_dir   = '/home/alex.piet/codebase/behavior/psy_fits_v9/'
t_model_dir = '/home/alex.piet/codebase/behavior/psy_fits_v10/'

# Get list of sessions
manifest = pgt.get_manifest()
ids = pgt.get_active_ids()

# Save results file for each session
po.build_all_session_outputs(ids,model_dir,output_dir):

# Build full summary table
po.build_summary_table(model_dir, output_dir)

# Build table of which models crashes
crash_manifest = po.build_list_of_model_crashes(model_dir)
a = crash_manifest.groupby(['stage']).agg({'model_fit':'sum')
b = crash_manifest.groupby(['stage']).agg({'model_fit':'size')
100*(a/b)

#### Training data
# Save results file for each non ophys training session, including habituation
train_manifest = pgt.get_training_manifest()
train_ids = train_manifest.query('not imaging').index.values
po.build_all_session_outputs(train_ids, t_model_dir, output_dir,TRAIN=True)

# Build summary table for training data
po.build_training_summary_table(t_model_dir, output_dir)

# Build table of which models crashed
crash_manifest = po.build_list_of_train_model_crashes(t_model_dir)
a = crash_manifest.groupby(['ophys','stage']).agg({'train_model_fit':'sum','ophys_model_fit':'sum'})
b = crash_manifest.groupby(['ophys','stage']).agg({'train_model_fit':'size','ophys_model_fit':'size'})
100*(a/b)

