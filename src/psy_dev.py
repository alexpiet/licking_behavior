import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
from importlib import reload
plt.ion()


directory="/home/alex.piet/codebase/behavior/psy_fits_v6/"
ids = ps.get_active_ids()

# Basic Characterization, Summaries at Session level
ps.plot_session_summary(ids,savefig=True,directory = directory)

# Basic Characterization, Summaries of each session
ps.summarize_fits(ids,directory)

ps.plot_fit(ids[0],directory=directory)

## PCA
###########################################################################################
drop_dex    = ps.PCA_dropout(ids,ps.get_mice_ids(),directory)
weight_dex  = ps.PCA_weights(ids,ps.get_mice_ids(),directory)
ps.PCA_analysis(ids, ps.get_mice_ids(),directory)

df = ps.get_all_timing_index(ids,directory)
ps.plot_model_index_summaries(df,directory)

## Clustering
###########################################################################################
# Get unified clusters
ps.build_all_clusters(ps.get_active_ids(), save_results=True)

## Compare fits. These comparisons are not exact, because some fits crashed on each version
########################################################################################### 
dir0 = "/home/alex.piet/codebase/behavior/psy_fits_v1/"
dir1 = "/home/alex.piet/codebase/behavior/psy_fits_v2/"
dir2 = "/home/alex.piet/codebase/behavior/psy_fits_v3_01/"
dir3 = "/home/alex.piet/codebase/behavior/psy_fits_v3_11/"
dir4 = "/home/alex.piet/codebase/behavior/psy_fits_v3/"
dir5 = "/home/alex.piet/codebase/behavior/psy_fits_v4/"
dir6 = "/home/alex.piet/codebase/behavior/psy_fits_v5/"
dir7 = "/home/alex.piet/codebase/behavior/psy_fits_v6/"
dirs = [dir0, dir1,dir2,dir3,dir4,dir5,dir6,dir7]

# Plot development history
all_roc = ps.compare_versions(dirs, ids)
ps.compare_versions_plot(all_roc)

# Comparing Timing versions
rocs = ps.compare_timing_versions(ids,"/home/alex.piet/codebase/behavior/psy_fits_v5/")

## Build Table of Mice by Strategy, cre line and depth
###########################################################################################
ps.build_manifest_by_task_index()


model_manifest = ps.build_model_manifest()
ps.plot_manifest_by_stage(model_manifest,'session_roc',hline=0.5,ylims=[0.5,1])
ps.plot_manifest_by_stage(model_manifest,'lick_fraction')
ps.plot_manifest_by_stage(model_manifest,'lick_hit_fraction')
ps.plot_manifest_by_stage(model_manifest,'trial_hit_fraction')

ps.plot_manifest_by_stage(model_manifest,'task_dropout_index')
ps.plot_manifest_by_stage(model_manifest,'task_weight_index')
ps.plot_manifest_by_stage(model_manifest,'prior_bias')
ps.plot_manifest_by_stage(model_manifest,'prior_task0')
ps.plot_manifest_by_stage(model_manifest,'prior_omissions1')
ps.plot_manifest_by_stage(model_manifest,'prior_timing1D')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_bias')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_task0')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_omissions1')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_timing1D')

# Looks like a real effect
ps.plot_manifest_by_stage(model_manifest,'task_dropout_index')
ps.plot_manifest_by_stage(model_manifest,'task_weight_index')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_task0')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_task0_1st')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_task0_2nd')

ps.plot_manifest_by_stage(model_manifest,'avg_weight_timing1D')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_timing1D_1st')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_timing1D_2nd')

ps.plot_manifest_by_stage(model_manifest,'avg_weight_bias')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_bias_1st')
ps.plot_manifest_by_stage(model_manifest,'avg_weight_bias_2nd')



# debugging double bout bug
##################
bad_cumsum = []
bad_cumsum2 = []
bad_flash = []
bad_flash2 = []
for index, id in enumerate(ids):
    session = ps.get_data(id)
    pm.annotate_licks(session) 
    pm.annotate_bouts(session)
    x = np.cumsum(session.stimulus_presentations.bout_start) - np.cumsum(session.stimulus_presentations.bout_end)    
    if np.any(x > 1):
        bad_cumsum.append(id)
        bad_flash.append(np.where(x>1)[0][0])
        plt.figure(); plt.plot(x)
    if np.any(x < 0):
        bad_cumsum2.append(id)
        bad_flash2.append(np.where(x<0)[0][0])   
        plt.figure(); plt.plot(x)


bad_ids = np.concatenate([bad_cumsum,bad_cumsum2])
bad_flash = np.concatenate([bad_flash, bad_flash2])

def analyze_bad_bout(id):
    session = ps.get_data(id)
    pm.annotate_licks(session) 
    pm.annotate_bouts(session)
    x = np.cumsum(session.stimulus_presentations.bout_start) - np.cumsum(session.stimulus_presentations.bout_end)    
    plt.figure()
    plt.plot(x)
    dex = np.where((x<0)|(x>1))[0][0]
    plt.plot(dex,x[dex],'ro')
    fit = ps.load_fit(id)
    return session,dex,fit
   
 
i+=1
id = bad_ids[i]
session,dex,fit = analyze_bad_bout(id)
psydata = ps.format_session(session,{})
prior_offset = 30
pt.plot_session(session)
dex_time = session.stimulus_presentations.iloc[dex].start_time
plt.gca().set_xlim(dex_time-prior_offset*.75, dex_time+10*0.75)

fit['psydata']['full_df'].iloc[dex-prior_offset:dex+10][['licked','bout_start','bout_end','in_bout']]
psydata['full_df'].iloc[dex-prior_offset:dex+10][['bout_start','bout_end','num_bout_start','num_bout_end','in_bout_raw','in_bout_raw_bad','in_bout']]


lens = []
old_lens = []
for index, id in enumerate(ids):
    print(id)
    session = ps.get_data(id)
    pm.annotate_licks(session)
    pm.annotate_bouts(session)
    psydata = ps.format_session(session,{})
    lens.append(len(psydata['y'])) 
    fit = ps.load_fit(id)
    old_lens.append(len(fit['psydata']['y']))


session2 = ps.get_data(id)
pt.annotate_licks(session2,bout_threshold=0.75) 
pm.annotate_bouts(session2)
plt.figure()
x2 = np.cumsum(session2.stimulus_presentations.bout_start) - np.cumsum(session2.stimulus_presentations.bout_end)
plt.plot(x2)
pt.plot_session(session2)
plt.gca().set_xlim(dex_time-prior_offset*.75, dex_time+10*0.75)



