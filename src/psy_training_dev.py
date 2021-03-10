import psy_tools as ps
import psy_general_tools as pgt
from alex_utils import *
import numpy as np
import pandas as pd
import psy_training_tools as ptt
import matplotlib.pyplot as plt
plt.ion()

# Build dataframe with results
train_summary = ps.build_model_training_manifest(version)

# Train Summary is a dataframe with model fit information
# Same as build_model_training_manifest, but loads from file??
train_summary = ptt.get_train_summary()

slc_train_summary = train_summary.query('cre_line == "Slc17a7-IRES2-Cre"').copy()
vip_train_summary = train_summary.query('cre_line == "Vip-IRES-Cre"').copy()
sst_train_summary = train_summary.query('cre_line == "Sst-IRES-Cre"').copy()

# Analysis Plots (Functions also accept metric='lick_hit_fraction')
ptt.plot_mouse_strategy_correlation(train_summary)
ptt.plot_strategy_correlation(train_summary)
ptt.plot_training(train_summary)
#ptt.plot_training_by_stage(train_summary)
#ptt.plot_training_dropout(train_summary)
#ptt.plot_training_roc(train_summary)

# new functions
ptt.plot_average_by_stage(training, metric='strategy_dropout_index')
ptt.plot_all_averages_by_stage(training,version)




