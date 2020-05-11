import psy_tools as ps
import psy_general_tools as pgt
from alex_utils import *
import numpy as np
import pandas pd
import psy_training_tools as ptt
import matplotlib.pyplot as plt
plt.ion()


training_manifest = pgt.get_training_manifest()

train_summary = ptt.get_train_summary()
ptt.plot_training_dropout(train_summary)
ptt.plot_training_roc(train_summary)


