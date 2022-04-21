import psy_output_tools as po
import psy_general_tools as pgt

# Look at a single session for a single version
bsid = 951520319
session = pgt.get_data(bsid) 
fit = ps.plot_fit(bsid,version=VERSION)

# Make a new version
VERSION = '21'
po.make_version(VERSION)

# Get directory for a version
directory = pgt.get_directory(VERSION) # main directory
figs_dir  = pgt.get_directory(VERSION, subdirectory='figures')
fits_dir  = pgt.get_directory(VERSION, subdirectory='fits')

# See what model versions are available
versions = po.get_model_versions(vrange=[20,25])

# Build inventory table
inventory_table = po.build_inventory_table(vrange=[20:25])

# Build summary files for each session, very slow
po.build_session_strategy_df(bsid, VERSION) 
po.build_session_strategy_df(bsid, VERSION, TRAIN=True)

# Build summary tables 
po.build_summary_table(VERSION)
po.build_training_summary_table(VERSION)# TODO Broken
po.build_mouse_summary_table(VERSION)   # TODO Broken

# Load summary tables
ophys_table    = po.get_ophys_summary_table(version)
training_table = po.get_training_summary_table(version)
mouse_table    = po.get_mouse_summary_table(version)


 


