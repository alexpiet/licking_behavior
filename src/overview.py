import psy_output_tools as po
import psy_general_tools as pgt

# Look at a single session for a single version
bsid = 951520319
session = pgt.get_data(bsid)
fit = ps.load_fit(bsid, version)
strategy_df = ps.load_session_strategy_df(bsid, version)
fit = ps.plot_fit(bsid,version=version)

# Make a new version
version = '21'
po.make_version(version)

# Get directory for a version
directory = pgt.get_directory(version) # main directory
figs_dir  = pgt.get_directory(version, subdirectory='figures')
fits_dir  = pgt.get_directory(version, subdirectory='fits')
stdf_dir  = pgt.get_directory(version, subdirectory='strategy_df')

# See what model versions are available
versions = po.get_model_versions(vrange=[20,25])

# Build inventory table
inventory_table = po.build_inventory_table(vrange=[20:25])

# Build strategy df files for a session. This is done when the
# model is fit, but if you want to do it manually. 
po.build_session_strategy_df(bsid, version) 
po.build_session_strategy_df(bsid, version, TRAIN=True)

# Build summary tables 
po.build_summary_table(version)
po.build_training_summary_table(version)# TODO Broken
po.build_mouse_summary_table(version)   # TODO Broken

# Load summary tables
ophys_table    = po.get_ophys_summary_table(version)
training_table = po.get_training_summary_table(version)
mouse_table    = po.get_mouse_summary_table(version)


 


