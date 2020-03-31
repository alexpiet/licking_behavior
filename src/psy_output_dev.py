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

po.build_all_session_outputs(ids,model_dir,output_dir):



