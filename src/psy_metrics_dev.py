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
pm.plot_metrics(session)        # plots metrics for this session
durations = pm.get_time_in_epochs(session) # Duration of each epoch

# Plot figure for each session
pm.plot_all_metrics(pgt.get_ophys_manifest())
pm.plot_all_metrics(pgt.get_training_manifest())

# Build summary df (VERY SLOW)
df = pm.build_metrics_df()
train_df = pm.build_metrics_df(TRAIN=True)

# get summary df
df = pm.get_metrics_df()
train_df = pm.get_metrics_df(TRAIN=True)

# Population Summary Figures
pm.plot_rates_summary(df)
pm.plot_rates_summary(df,group='cre_line')
pm.plot_rates_summary(df,group='session_type')

pm.plot_count_summary(df)
pm.plot_count_summary(df, group='cre_line')
pm.plot_count_summary(df, group='session_type')
pm.plot_counts(df, ['fraction_low_lick_low_reward','fraction_high_lick_high_reward','fraction_high_lick_low_reward'] ,ylim=(0,1),label='epoch')

### TODO
# improve color/linestyle definitions
# Add significance testing




