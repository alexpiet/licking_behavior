import psy_output_tools as po
import psy_general_tools as pgt

## Single Session
################################################################################
# Look at a single session for a single version
bsid = 951520319
session = pgt.get_data(bsid)

# load fit for a single session
fit = ps.load_fit(bsid, version)
strategy_df = ps.load_session_strategy_df(bsid, version)
fit = ps.plot_fit(bsid,version=version)

# Build strategy df files for a session. This is done when the
# model is fit, but if you want to do it manually. 
ps.build_session_strategy_df(bsid, version) 
ps.build_session_strategy_df(bsid, version, TRAIN=True)
strategy_fit = ps.load_session_strategy_df(bsid, version):

## Versions
################################################################################
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
inventory_table = po.build_inventory_table(vrange=[20,25])

# Build summary tables 
po.build_summary_table(version)
po.build_training_summary_table(version)# TODO Broken
po.build_mouse_summary_table(version)   # TODO Broken

# Load summary tables
ophys_df    = po.get_ophys_summary_table(version)
training_df = po.get_training_summary_table(version) # TODO Broken
mouse_df    = po.get_mouse_summary_table(version)    # TODO Broken

## Analysis
################################################################################
summary_df = po.get_ophys_summary_df(version)

pv.plot_session_summary(summary_df,version=version)






