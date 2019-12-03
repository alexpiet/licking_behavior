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

# Get Some Sessions to analyze
manifest = ps.get_manifest()
active_slc = np.intersect1d(ps.get_slc_session_ids(),ps.get_active_ids())
passive_slc = np.intersect1d(ps.get_slc_session_ids(),ps.get_passive_ids())
depth175 = ps.get_layer_ids(175)
depth375 = ps.get_layer_ids(375)
session_list = np.intersect1d(active_slc,depth175)

# Test, comparing df and list implementations
test_df, test_cell_cms, test_cell_mean_cms, test_cell_var_cms, test_session_means,test_session_vars, test_pop_mean,test_pop_var = pg.manifest_change_modulation(session_list[0:2])
pg.plot_manifest_change_modulation(test_cell_cms,test_cell_mean_cms,test_session_means,plot_cells=False)
pg.plot_manifest_change_modulation_df(test_df,plot_cells=False)
pg.plot_manifest_change_modulation(test_cell_cms,test_cell_mean_cms,test_session_means,plot_cells=True)
pg.plot_manifest_change_modulation_df(test_df,plot_cells=True)
pg.plot_manifest_change_modulation_df(test_df,plot_cells=False,metric='change_modulation_base')
pg.plot_manifest_change_modulation_df(test_df,plot_cells=True,metric='change_modulation_base')

# Plot single session
single_df, *single_list = pg.manifest_change_modulation(session_list[0:1])
pg.plot_manifest_change_modulation_df(single_df)
pg.plot_manifest_change_modulation_df(single_df,metric='change_modulation_base')

# Do Single session bootstrapping and compare distributions
boot_df = pg.bootstrap_session_cell_modulation_df(single_df,15)
pg.plot_manifest_change_modulation_df(boot_df)
pg.compare_dist_df([boot_df,single_df],[20,20],['k','r'],['Shuffle','Data'],[1,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
pg.compare_dist_df([boot_df,single_df],[20,20],['k','r'],['Shuffle','Data'],[1,0.5],ylabel='Prob/Bin',xlabel='Change Modulation Base',metric='change_modulation_base')

# Testing effects of removing unreliable cells
test_df_unreliable, *test_list_unreliable = pg.manifest_change_modulation(session_list[0:1],remove_unreliable=False)
test_df_reliable,   *test_list_reliable   = pg.manifest_change_modulation(session_list[0:1],remove_unreliable=True)
pg.plot_manifest_change_modulation_df(test_df_unreliable)
pg.plot_manifest_change_modulation_df(test_df_reliable)
pg.compare_dist_df([test_df_unreliable,test_df_reliable],[20,20],['k','r'],['unreliable','reliable'],[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
pg.compare_dist_df([test_df_unreliable,test_df_reliable],[20,20],['k','r'],['unreliable','reliable'],[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation Base',metric='change_modulation_base')

# get everything (SLOW to compute, fast to load from disk)
if False:
    all_df, *all_list = pg.manifest_change_modulation(ps.get_slc_session_ids())
    all_df = pg.annotate_stage(all_df)
    all_df.to_csv(path_or_buf='/home/alex.piet/Desktop/all_slc_df.csv')
else:
    all_df =pd.read_csv(filepath_or_buffer = '/home/alex.piet/Desktop/all_slc_df.csv')

pg.plot_manifest_change_modulation_df(all_df,plot_cells=False)

# compare active passive
pg.compare_groups_df([all_df.query('active'),all_df.query('not active')],['Active', 'Passive'],savename="all_active_passive")
pg.compare_groups_df([all_df.query('active & imaging_depth == 175'),all_df.query('not active & imaging_depth == 175 ')],['Active 175', 'Passive 175'],savename="active_passive_175")
pg.compare_groups_df([all_df.query('active & imaging_depth == 375'),all_df.query('not active & imaging_depth == 375 ')],['Active 375', 'Passive 375'],savename="active_passive_375")

# compare A/B
pg.compare_groups_df([all_df.query('image_set == "A"'),all_df.query('image_set == "B"')],['A', 'B'],savename="all_A_B")

# compare A/B and active/passive
pg.compare_groups_df([all_df.query('image_set == "A" & active'),all_df.query('image_set == "A" & not active')],['Active A', 'Passive A'],savename="active_passive_A")
pg.compare_groups_df([all_df.query('image_set == "B" & active'),all_df.query('image_set == "B" & not active')],['Active B', 'Passive B'],savename="active_passive_B")

pg.compare_groups_df([all_df.query('image_set == "A" & active'),all_df.query('image_set == "B" & active')],['Active A', 'Active B'],savename="active_A_B")
pg.compare_groups_df([all_df.query('image_set == "A" & not active'),all_df.query('image_set == "B" & not active')],['Passive A', 'Passive B'],savename="passive_A_B")


# plot 175/375mm depth SLC active/passive A/B Images
pg.compare_groups_df([all_df.query('active & imaging_depth == 175 & image_set == "A"'),all_df.query('not active & imaging_depth == 175 & image_set == "A"')],['Active 175 A', 'Passive 175 A'],savename="active_passive_175_A")
pg.compare_groups_df([all_df.query('active & imaging_depth == 175 & image_set == "B"'),all_df.query('not active & imaging_depth == 175 & image_set == "B"')],['Active 175 B', 'Passive 175 B'],savename="active_passive_175_B")
pg.compare_groups_df([all_df.query('active & imaging_depth == 375 & image_set == "A"'),all_df.query('not active & imaging_depth == 375 & image_set == "A"')],['Active 375 A', 'Passive 375 A'],savename="active_passive_375_A")
pg.compare_groups_df([all_df.query('active & imaging_depth == 375 & image_set == "B"'),all_df.query('not active & imaging_depth == 375 & image_set == "B"')],['Active 375 B', 'Passive 375 B'],savename="active_passive_375_B")

# Stage 3/4 Comparisons
pg.compare_groups_df([all_df.query('imaging_depth == 175 & stage_num == "3"'),all_df.query('imaging_depth == 175 & stage_num == "4"')],['175 Stage 3','175 Stage 4'],savename="by_stage34_175")
pg.compare_groups_df([all_df.query('imaging_depth == 375 & stage_num == "3"'),all_df.query('imaging_depth == 375 & stage_num == "4"')],['375 Stage 3','375 Stage 4'],savename="by_stage34_375")

# Stage 4/6 Comparisons
pg.compare_groups_df([all_df.query('stage_num == "4"'),all_df.query('stage_num == "6"')],['Stage 4','Stage 6'],savename="by_stage46")
pg.compare_groups_df([all_df.query('imaging_depth == 175 & stage_num == "4"'),all_df.query('imaging_depth == 175 & stage_num == "6"')],['175 Stage 4','175 Stage 6'],savename="by_stage46_175")
pg.compare_groups_df([all_df.query('imaging_depth == 375 & stage_num == "4"'),all_df.query('imaging_depth == 375 & stage_num == "6"')],['375 Stage 4','375 Stage 6'],savename="by_stage46_375")


# compute trianges of variance
# 175 Depth
session_var = np.mean(session_vars)
cell_var = np.mean(np.hstack(cell_var_cms))
all_vars = [cell_var, session_var, pop_var]
all_vars_norm = all_vars/np.sum(all_vars)
# 375 Depth
session_var3 = np.mean(session_vars3)
cell_var3 = np.mean(np.hstack(cell_var_cms3))
all_vars3 = [cell_var3, session_var3, pop_var3]
all_vars_norm3 = all_vars3/np.sum(all_vars3)

## Include VIP for comparisons
vip = [0.96,0.035,0.0038]
vip_var = 0.1
total_vars = [np.sum(all_vars)/pop_mean, np.sum(all_vars3)/pop_mean3, 0.686]
pg.plot_simplex([all_vars_norm,all_vars_norm3,vip],['Flashes','Cells','Sessions'],['175 Slc','375 Slc','VIP'],['r','r','b'],total_vars)

