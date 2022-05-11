import numpy as np
import psy_general_tools as pgt
import matplotlib.pyplot as plt
import psy_metrics_tools as pm
import pandas as pd
from scipy import stats
import seaborn as sns
plt.ion()

# Basic Example
ophys = pgt.get_ophys_manifest()
bsid = ophys['behavior_session_ids'].values[0]
session = pgt.get_data(bsid)    # Get SDK session object
pm.get_metrics(session)         # annotate session




