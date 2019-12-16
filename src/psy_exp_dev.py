import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
from importlib import reload
plt.ion()
import numpy as np
import pandas as pd
import psy_glm_tools as pg
import hierarchical_boot as hb
from tqdm import tqdm
import seaborn as sns


all_df = get_all_df()
compare_groups(all_df,['active','not active'],['Active','Passive'])
compare_groups(all_df,['active & imaging_depth == 175','not active & imaging_depth == 175'],['Active 175','Passive 175'])
compare_groups(all_df,['active & imaging_depth == 375','not active & imaging_depth == 375'],['Active 375','Passive 375'])

compare_groups(all_df,['image_set == "A"','image_set == "B"'],['A','B'])

compare_groups(all_df,['stage_num == "1"','stage_num == "4"'],['1','4'])
compare_groups(all_df,['stage_num == "4"','stage_num == "6"'],['4','6'])

plt.figure()
cm_4_175 = (np.mean(cell_psths_4_175,0)-np.mean(cell_psths_6_175,0))/(np.mean(cell_psths_4_175,0)+np.mean(cell_psths_6_175,0))
cm_4_375 = (np.mean(cell_psths_4_375,0)-np.mean(cell_psths_6_375,0))/(np.mean(cell_psths_4_375,0)+np.mean(cell_psths_6_375,0))
plt.plot(cm_4_175,color=colors[0],label='CM 175')
plt.plot(cm_4_375,color=colors[1],label='CM 375')






