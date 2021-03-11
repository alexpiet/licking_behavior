import numpy as np
import pandas as pd
import psy_tools as ps
import psy_general_tools as pgt
import psy_training_tools as ptt
import matplotlib.pyplot as plt
import psy_output_tools as po
plt.ion()

# Train Summary is a dataframe with model fit information
train_summary = po.get_training_summary_table()

# Plot Averages by training stage 
ptt.plot_average_by_stage(training, metric='strategy_dropout_index')
ptt.plot_all_averages_by_stage(training,version)
ptt.plot_all_averages_by_stage(training,version,plot_mouse_groups=True)
ptt.plot_all_averages_by_stage(training,version,plot_each_mouse=True)


##### DEV BELOW HERE
# Analysis Plots (Functions also accept metric='lick_hit_fraction')
ptt.plot_mouse_strategy_correlation(train_summary)
ptt.plot_strategy_correlation(train_summary)
ptt.plot_training(train_summary)
#ptt.plot_training_by_stage(train_summary)
#ptt.plot_training_dropout(train_summary)
#ptt.plot_training_roc(train_summary)




