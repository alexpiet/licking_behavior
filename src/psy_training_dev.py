import psy_tools as ps
import psy_general_tools as pgt
from alex_utils import *
import numpy as np
import pandas as pd
import psy_training_tools as ptt
import matplotlib.pyplot as plt
plt.ion()


# Training Manifest is a dataframe of sessions
training_manifest = pgt.get_training_manifest()

# Train Summary is a dataframe with model fit information
train_summary = ptt.get_train_summary()

# Analysis Plots
ptt.plot_strategy_correlation(train_summary)
ptt.plot_training(train_summary)
ptt.plot_training_dropout(train_summary)
ptt.plot_training_roc(train_summary)


