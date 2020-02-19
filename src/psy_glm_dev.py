import numpy as np
import pandas as pd
import psy_tools as ps
import psy_cluster as pc
import psy_glm_tools as pg
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import hierarchical_boot as hb
import psy_general_tools as pgt
from tqdm import tqdm
from alex_utils import *
from importlib import reload
plt.ion()

# Get Some Sessions to analyze
manifest = pgt.get_manifest()
active_slc = np.intersect1d(pgt.get_slc_session_ids(),pgt.get_active_ids())
passive_slc = np.intersect1d(pgt.get_slc_session_ids(),pgt.get_passive_ids())
depth175 = pgt.get_layer_ids(175)
depth375 = pgt.get_layer_ids(375)
active_slc_175 = np.intersect1d(active_slc,depth175)

# Test, comparing df and list implementations
test_df, test_cell_cms, test_cell_mean_cms, test_cell_var_cms, 
    test_session_means,test_session_vars, test_pop_mean,test_pop_var = 
    pg.manifest_change_modulation(active_slc_175[0:2])
pg.plot_manifest_change_modulation(test_cell_cms,test_cell_mean_cms, 
    test_session_means,plot_cells=False)
pg.plot_manifest_change_modulation_df(test_df,plot_cells=False)
pg.plot_manifest_change_modulation(test_cell_cms,test_cell_mean_cms, 
    test_session_means,plot_cells=True)
pg.plot_manifest_change_modulation_df(test_df,plot_cells=True)
pg.plot_manifest_change_modulation_df(test_df,plot_cells=False, metric='change_modulation_base')
pg.plot_manifest_change_modulation_df(test_df,plot_cells=True, metric='change_modulation_base')

# Plot single session
single_df, *single_list = pg.manifest_change_modulation(active_slc_175[0:1])
single_df_f = single_df.query('reliable_cell & real_response & good_response & good_block')
pg.plot_manifest_change_modulation_df(single_df_f)
pg.plot_manifest_change_modulation_df(single_df_f,metric='change_modulation_base')
pg.plot_manifest_change_modulation_df(single_df_f,metric='change_modulation_dc')
pg.plot_manifest_change_modulation_df(single_df_f,metric='change_modulation_base_dc')
pg.compare_groups_df([single_df_f],['example session'])
pg.plot_top_cell(single_df_f)
pg.plot_top_cell(single_df_f,top=-1)
pg.plot_top_cell(single_df_f,show_all_blocks=True)
pg.compare_groups_df([single_df_f],['example session'],metric='change_modulation_dc')
pg.plot_top_cell(single_df_f,metric='change_modulation_dc')
pg.plot_top_cell(single_df_f,top=-1,metric='change_modulation_dc')
pg.plot_top_n_cells(single_df_f,n=144)

# Do Single session bootstrapping and compare distributions
boot_df = pg.bootstrap_session_cell_modulation_df(single_df,15)
pg.plot_manifest_change_modulation_df(boot_df)
pg.compare_dist_df([boot_df,single_df],[20,20],['k','r'],['Shuffle','Data'],
    [1,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
pg.compare_dist_df([boot_df,single_df],[20,20],['k','r'],['Shuffle','Data'],
    [1,0.5],ylabel='Prob/Bin',xlabel='Change Modulation Base',metric='change_modulation_base')

# Testing effects of removing unreliable cells
test_df_unreliable, *test_list_unreliable = pg.manifest_change_modulation(active_slc_175[0:1],remove_unreliable=False)
test_df_reliable, *test_list_reliable = pg.manifest_change_modulation(active_slc_175[0:1],remove_unreliable=True)
pg.plot_manifest_change_modulation_df(test_df_unreliable)
pg.plot_manifest_change_modulation_df(test_df_reliable)
pg.compare_dist_df([test_df_unreliable,test_df_reliable],[20,20],['k','r'],
    ['unreliable','reliable'],[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
pg.compare_dist_df([test_df_unreliable,test_df_reliable],[20,20],['k','r'],
    ['unreliable','reliable'],[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation Base',
    metric='change_modulation_base')

# Demonstration of multiple sessions
two_df, *two_list = pg.manifest_change_modulation(active_slc_175[0:2])
two_df_f = two_df.query('reliable_cell & real_response & good_response')
pg.plot_manifest_change_modulation_df(two_df_f)
pg.plot_manifest_change_modulation_df(two_df_f,metric='change_modulation_base')
pg.plot_manifest_change_modulation_df(two_df_f,metric='change_modulation_dc')
pg.plot_manifest_change_modulation_df(two_df_f,metric='change_modulation_base_dc')
pg.compare_groups_df([two_df_f],['two examples'])
pg.compare_groups_df([two_df_f],['two examples'],metric='change_modulation_dc')

# Compute a few more for debugging
many_df = pg.get_all_df(ids=active_slc_175[0:10],force_recompute=True, savefile=False)
many_df_f = many_df.query('reliable_cell & real_response & good_response & good_block')
pg.plot_manifest_change_modulation_df(many_df_f,plot_cells=False)
pg.compare_groups_df([many_df_f],['example session'])
pg.plot_top_cell(many_df_f)
pg.plot_top_cell(many_df_f,top=-1)






import seaborn as sns
sns.set_context('notebook', font_scale=1, rc={'lines.markeredgewidth': 2})
sns.set_style('white', {'axes.spines.right': False, 'axes.spines.top': False, 'xtick.bottom': False, 'ytick.left': False,})


#######################################################################
# get everything (SLOW to compute, fast to load from disk)
all_df = pg.get_all_df()

# Append Cre line
experiment_table = pgt.get_experiment_table()
df = pd.DataFrame()
df['ophys_experiment_id'] = experiment_table.reset_index()['ophys_experiment_id']
df['cre_line'] = [x[0:3] for x in experiment_table.full_genotype]
all_df = pd.merge(all_df,df,on='ophys_experiment_id')

# Filter cells/responses
all_df_f1 = all_df.query('reliable_cell & real_response & good_response & good_block')

# Append Number blocks and filter
all_df_f1 = pg.add_num_blocks(all_df_f1)
all_df_f = all_df_f1.query('num_blocks > 5')




#######################################################################
# Simple Analysis
pg.plot_manifest_change_modulation_df(all_df_f,plot_cells=False)
pg.plot_top_n_cells(all_df_f,n=10)
pg.plot_top_n_cells(all_df_f,n=-1)

# Compare Cre lines
pg.compare_groups_df([all_df_f.query('cre_line=="Slc"')], ['Slc'],savename="all_Slc",nboots=100)
pg.compare_groups_df([all_df_f.query('cre_line=="Slc"')], ['Slc'],savename="all_Slc",metric='change_modulation_dc',nboots=100,nbins=[3,50,100],ylim=(-.25,.15),plot_nice=True, ci=True)
pg.compare_groups_df([all_df_f.query('cre_line=="Vip"')], ['Vip'],savename="all_Vip",nboots=100)
pg.compare_groups_df([all_df_f.query('cre_line=="Vip"')], ['Vip'],savename="all_Vip",metric='change_modulation_dc',nboots=100)
pg.compare_groups_df([all_df_f.query('cre_line=="Sst"')], ['Sst'],savename="all_Sst",nboots=100)
pg.compare_groups_df([all_df_f.query('cre_line=="Sst"')], ['Sst'],savename="all_Sst",metric='change_modulation_dc',nboots=100)
pg.compare_groups_df([all_df_f.query('cre_line=="Slc"'),all_df_f.query('cre_line =="Vip"'), 
                        all_df_f.query('cre_line=="Sst"')], ['Slc','Vip','Sst'],savename="all_cre_line")
pg.compare_groups_df([all_df_f.query('cre_line=="Slc"'),all_df_f.query('cre_line =="Vip"'), 
                        all_df_f.query('cre_line=="Sst"')], ['Slc','Vip','Sst'],savename="all_cre_line",
                        metric='change_modulation_dc',nbins=[[3,3,3],[50,10,10],[100,25,25]],ylim=(-.25,.15),plot_nice=True)

# Compare Active/Passive
pg.compare_groups_df(   [all_df_f.query('cre_line=="Slc" & active'),
                        all_df_f.query('cre_line=="Slc" & not active')], 
                        ['Slc active','Slc passive'],savename="all_Slc_active_passive")
pg.compare_groups_df(   [all_df_f.query('cre_line=="Vip" & active'),
                        all_df_f.query('cre_line=="Vip" & not active')], 
                        ['Vip active','Vip passive'],savename="all_Vip_active_passive")
pg.compare_groups_df(   [all_df_f.query('cre_line=="Sst" & active'),all_df_f.query('cre_line=="Sst" & not active')], 
                        ['Sst active','Sst passive'],savename="all_Sst_active_passive")

pg.compare_groups_df(   [all_df_f.query('cre_line=="Slc" & active'),all_df_f.query('cre_line=="Slc" & not active')], 
                        ['Slc active','Slc passive'],savename="all_Slc_active_passive",metric='change_modulation_dc',
                        nbins=[[3,3],[50,50],[100,100]],ylim=(-.25,.15),plot_nice=True)
pg.compare_groups_df(   [all_df_f.query('cre_line=="Vip" & active'),all_df_f.query('cre_line=="Vip" & not active')], 
                        ['Vip active','Vip passive'],savename="all_Vip_active_passive",metric='change_modulation_dc',
                        nbins=[[3,3],[10,10],[25,25]],ylim=(-.6,.05),plot_nice=True)
pg.compare_groups_df(   [all_df_f.query('cre_line=="Sst" & active'),all_df_f.query('cre_line=="Sst" & not active')], 
                        ['Sst active','Sst passive'],savename="all_Sst_active_passive",metric='change_modulation_dc',
                        nbins=[[3,3],[10,10],[25,25]],ylim=(-.4,.05),plot_nice=True)

# Compare Imaging Depth 
pg.compare_groups_df([all_df_f.query('cre_line=="Slc" & imaging_depth==175'),
                    all_df_f.query('cre_line=="Slc" & imaging_depth==375')], 
                    ['Slc 175','Slc 375'],savename="all_Slc_depth")
pg.compare_groups_df([all_df_f.query('cre_line=="Slc" & imaging_depth==175'),
                    all_df_f.query('cre_line=="Slc" & imaging_depth==375')], 
                    ['Slc 175','Slc 375'],savename="all_Slc_depth",metric='change_modulation_dc',
                    nbins=[[3,3],[50,50],[100,100]],ylim=(-.25,.15),plot_nice=True)

# Compare Imaging Depth X Active/Passive
pg.compare_groups_df(   [all_df_f.query('cre_line=="Slc" & active & imaging_depth==175'),
                        all_df_f.query('cre_line=="Slc" & not active & imaging_depth==175')], 
                        ['Slc active 175','Slc passive 175'],savename="all_Slc_active_passive_175")
pg.compare_groups_df(   [all_df_f.query('cre_line=="Slc" & active & imaging_depth==175'),
                        all_df_f.query('cre_line=="Slc" & not active & imaging_depth==175')], 
                        ['Slc active 175','Slc passive 175'],savename="all_Slc_active_passive_175",
                        metric='change_modulation_dc')
pg.compare_groups_df(   [all_df_f.query('cre_line=="Slc" & active & imaging_depth==375'),
                        all_df_f.query('cre_line=="Slc" & not active & imaging_depth==375')], 
                        ['Slc active 375','Slc passive 375'],savename="all_Slc_active_passive_375")
pg.compare_groups_df(   [all_df_f.query('cre_line=="Slc" & active & imaging_depth==375'),
                        all_df_f.query('cre_line=="Slc" & not active & imaging_depth==375')], 
                        ['Slc active 375','Slc passive 375'],savename="all_Slc_active_passive_375",
                        metric='change_modulation_dc')






# compare A/B
pg.compare_groups_df(
    [all_df_f.query('image_set == "A"'),
    all_df_f.query('image_set == "B"')],['A', 'B'],savename="all_A_B")

# compare A/B and active/passive
pg.compare_groups_df([
    all_df_f.query('image_set == "A" & active'),
    all_df_f.query('image_set == "A" & not active')],
    ['Active A', 'Passive A'],savename="active_passive_A")
pg.compare_groups_df([
    all_df_f.query('image_set == "B" & active'),
    all_df_f.query('image_set == "B" & not active')],
    ['Active B', 'Passive B'],savename="active_passive_B")
pg.compare_groups_df([
    all_df_f.query('image_set == "A" & active'),
    all_df_f.query('image_set == "B" & active')],
    ['Active A', 'Active B'],savename="active_A_B")
pg.compare_groups_df(
    [all_df_f.query('image_set == "A" & not active'),
    all_df_f.query('image_set == "B" & not active')],
    ['Passive A', 'Passive B'],savename="passive_A_B")

# plot 175/375mm depth SLC active/passive A/B Images
pg.compare_groups_df([
    all_df_f.query('active & imaging_depth == 175 & image_set == "A"'),
    all_df_f.query('not active & imaging_depth == 175 & image_set == "A"')],
    ['Active 175 A', 'Passive 175 A'],savename="active_passive_175_A")
pg.compare_groups_df([
    all_df_f.query('active & imaging_depth == 175 & image_set == "B"'),
    all_df_f.query('not active & imaging_depth == 175 & image_set == "B"')],
    ['Active 175 B', 'Passive 175 B'],savename="active_passive_175_B")
pg.compare_groups_df([
    all_df_f.query('active & imaging_depth == 375 & image_set == "A"'),
    all_df_f.query('not active & imaging_depth == 375 & image_set == "A"')],
    ['Active 375 A', 'Passive 375 A'],savename="active_passive_375_A")
pg.compare_groups_df([
    all_df_f.query('active & imaging_depth == 375 & image_set == "B"'),
    all_df_f.query('not active & imaging_depth == 375 & image_set == "B"')],
    ['Active 375 B', 'Passive 375 B'],savename="active_passive_375_B")

# Stage 3/4 Comparisons
pg.compare_groups_df([
    all_df_f.query('imaging_depth == 175 & stage_num == "3"'),
    all_df_f.query('imaging_depth == 175 & stage_num == "4"')],
    ['175 Stage 3','175 Stage 4'],savename="by_stage34_175")
pg.compare_groups_df(
    [all_df_f.query('imaging_depth == 375 & stage_num == "3"'),
    all_df_f.query('imaging_depth == 375 & stage_num == "4"')],
    ['375 Stage 3','375 Stage 4'],savename="by_stage34_375")

# Stage 4/6 Comparisons
pg.compare_groups_df(
    [all_df_f.query('stage_num == "4"'),all_df_f.query('stage_num == "6"')],
    ['Stage 4','Stage 6'],savename="by_stage46")
pg.compare_groups_df(
    [all_df_f.query('imaging_depth == 175 & stage_num == "4"'),
    all_df_f.query('imaging_depth == 175 & stage_num == "6"')],
    ['175 Stage 4','175 Stage 6'],savename="by_stage46_175")
pg.compare_groups_df(
    [all_df_f.query('imaging_depth == 375 & stage_num == "4"'),
    all_df_f.query('imaging_depth == 375 & stage_num == "6"')],
    ['375 Stage 4','375 Stage 6'],savename="by_stage46_375")

# Comparing Depth
pg.compare_groups_df(
    [all_df_f.query('imaging_depth == 175'),
    all_df_f.query('imaging_depth == 375')],
    ['175', '375'],savename="all_175_375")

# compute trianges of variance
var_vec, ff = pg.get_variance_by_level(all_df_f)
varA,ffA = pg.get_variance_by_level(all_df_f.query('image_set == "A"'))
varB,ffB = pg.get_variance_by_level(all_df_f.query('image_set == "B"'))
var1,ff1 = pg.get_variance_by_level(all_df_f.query('imaging_depth == 175'))
var3,ff3 = pg.get_variance_by_level(all_df_f.query('imaging_depth == 375'))

pg.plot_simplex([var_vec],['Flashes','Cells','Sessions'],['All'],['k'],[ff])
pg.plot_simplex([varA, varB],['Flashes','Cells','Sessions'],['A','B'],['r','b'],[ffA,ffB])
pg.plot_simplex([var1,var3],['Flashes','Cells','Sessions'],['175','375'],['g','m'],[ff1,ff3])


# White Paper plots
pg.compare_groups_df([all_df_f.query('imaging_depth == 175 & stage_num == "1"'),
    all_df_f.query('imaging_depth == 175 & stage_num == "2"')],['175 A1','175 A2'],
    savename="by_A1_A2_175",plot_nice=True,nboots=10000)
pg.compare_groups_df([all_df_f.query('imaging_depth == 375 & stage_num == "1"'),
    all_df_f.query('imaging_depth == 375 & stage_num == "2"')],['375 A1','375 A2'],
    savename="by_A1_A2_375",plot_nice=True,nboots=10000)
pg.compare_groups_df([all_df_f.query('imaging_depth == 175 & stage_num == "4"'),
    all_df_f.query('imaging_depth == 175 & stage_num == "5"')],['175 B1','175 B2'],
    savename="by_B4_B5_175",plot_nice=True,nboots=10000)
pg.compare_groups_df([all_df_f.query('imaging_depth == 375 & stage_num == "4"'),
    all_df_f.query('imaging_depth == 375 & stage_num == "5"')],['375 B1','375 B2'],
    savename="by_B4_B5_375",plot_nice=True,nboots=10000)

# Have to update function call like this
pg.compare_groups_df([
 slc_df.query('good_response & good_block & reliable_cell & imaging_depth==175 & stage_num=="1"'),
 slc_df.query('good_response & good_block & reliable_cell & imaging_depth==175 & stage_num=="2"')],
 ['175 A1','175 A2'],savename="test",plot_nice=True,nboots=100)


#### Full trace
all_exp_df = pg.get_all_exp_df()

# Query a specific session
pg.plot_top_cell(all_df.query('ophys_experiment_id == @session_id[0]'),all_exp_df,'')

# Find Top across all sessions
pg.plot_top_cell(all_df,all_exp_df,'')


pg.compare_exp_groups(all_exp_df,['active','not active'],['Active','Passive'])
pg.compare_exp_groups(all_exp_df,
    ['active & imaging_depth == 175','not active & imaging_depth == 175'],['Active 175','Passive 175'])
pg.compare_exp_groups(all_exp_df,
    ['active & imaging_depth == 375','not active & imaging_depth == 375'],['Active 375','Passive 375'])
pg.compare_exp_groups(all_exp_df,
    ['image_set=="A" & active & imaging_depth == 175',
    'image_set=="A" & not active & imaging_depth == 175'],['A Active 175',' A Passive 175'])
pg.compare_exp_groups(all_exp_df,
    ['image_set=="B" & active & imaging_depth == 175',
    'image_set=="B" & not active & imaging_depth == 175'],['B Active 175',' B Passive 175'])
pg.compare_exp_groups(all_exp_df,
    ['image_set=="A" & active & imaging_depth == 375',
    'image_set=="A" & not active & imaging_depth == 375'],['A Active 375',' A Passive 375'])
pg.compare_exp_groups(all_exp_df,
    ['image_set=="B" & active & imaging_depth == 375',
    'image_set=="B" & not active & imaging_depth == 375'],['B Active 375',' B Passive 375'])

pg.compare_exp_groups(all_exp_df,['image_set == "A"','image_set == "B"'],['A','B'])
pg.compare_exp_groups(all_exp_df,['stage_num == "1"','stage_num == "4"'],['1','4'])
pg.compare_exp_groups(all_exp_df,['stage_num == "4"','stage_num == "6"'],['4','6'])

##################################

slc_df, slc_cell_df = pg.get_slc_dfs()
top_cells = pg.get_top_cells(slc_df, 10,metric='change_modulation_dc',query='good_response_dc & reliable_cell & good_block & real_response')
pg.plot_mean_trace(slc_df.query('cell ==@top_cells[0]'),'good_block')
pg.plot_mean_trace(slc_df.query('cell ==@top_cells[0]'),'good_response_dc& good_block')


