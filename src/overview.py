import psy_output_tools as po
import psy_general_tools as pgt

# Make a new version
VERSION = '21'
po.make_version(VERSION)

# Get directory for a version
directory = pgt.get_directory(VERSION) # main directory
figs_dir = pgt.get_directory(VERSION, subdirectory='figures')
fits_dir = pgt.get_directory(VERSION, subdirectory='fits')

# See What model versions are available
versions = po.get_model_versions(vrange=[20,25])

# Build inventory table
inventory_table = po.build_inventory_table(vrange=[20:25])

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

 


