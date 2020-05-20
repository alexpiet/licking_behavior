import psy_general_tools as pgt
import psy_metrics_tools as pm
import psy_tools as ps
import psy_output_tools as po
from alex_utils import *


# QC dev
train_summary = pd.read_csv(output_dir+"_summary_table.csv")
no_hits = train_summary.query('num_hits =="0"')

#
# Define model version, and output directory
output_dir  = '/home/alex.piet/codebase/behavior/model_output/'
model_dir   = '/home/alex.piet/codebase/behavior/psy_fits_v10/'
t_model_dir = '/home/alex.piet/codebase/behavior/psy_fits_v10/'

# Build full summary table
po.build_summary_table(model_dir, output_dir)

# Get list of sessions
ids = pd.read_csv(output_dir+'_summary_table.csv')['behavior_session_id'].values

# Save results file for each session
po.build_all_session_outputs(ids,model_dir,output_dir,hit_threshold=0):

# Build table of which models crashes
crash_manifest = po.build_list_of_model_crashes(model_dir)
a = crash_manifest.groupby(['stage']).agg({'model_fit':'sum')
b = crash_manifest.groupby(['stage']).agg({'model_fit':'size')
100*(a/b)

#### Training data
# Build summary table for training data
po.build_training_summary_table(t_model_dir, output_dir,hit_threshold=0)

# Save results file for each non ophys training session, including habituation
train_ids = pd.read_csv(output_dir+"_training_summary_table.csv")['behavior_session_id'].values
po.build_all_session_outputs(train_ids, t_model_dir, output_dir,TRAIN=True)

# Build table of which models crashed
crash_manifest = po.build_list_of_train_model_crashes(t_model_dir)
a = crash_manifest.groupby(['ophys','stage']).agg({'train_model_fit':'sum','ophys_model_fit':'sum'})
b = crash_manifest.groupby(['ophys','stage']).agg({'train_model_fit':'size','ophys_model_fit':'size'})
100*(a/b)

