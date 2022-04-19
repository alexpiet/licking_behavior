import psy_output_tools as po
import psy_general_tools as pgt

# Look at a single session for a single version
bsid = 951520319
session = pgt.get_data(bsid) # TODO Broken
fit = ps.plot_fit(bsid,version=VERSION)

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

# Build summary files for each session, very slow
po.build_session_output(bsid, VERSION) 
po.build_session_output(bsid, VERSION, TRAIN=True)
po.build_all_session_outputs(VERSION) 
po.build_all_session_outputs(VERSION,TRAIN=True)

# Build summary tables 
po.build_summary_table(VERSION)
po.build_training_summary_table(VERSION) #TODO Broken
po.build_mouse_summary_table(VERSION) #TODO Broken

# Load summary tables
ophys_table = po.get_ophys_summary_table(version)
training_table = po.get_training_summary_table(version)
mouse_table = po.get_mouse_summary_table(version)

# Get a table of sessions without model fits #TODO, should merge into inventory tabl 
crash_manifest = po.build_list_of_model_crashes(VERSION)
crash_manifest = po.build_list_of_train_model_crashes(VERSION)

 


