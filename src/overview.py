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




