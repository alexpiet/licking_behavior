import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
from importlib import reload
plt.ion()

import psy_glm_tools as pg

# run one session #1,17 good test sessions
ids = ps.get_session_ids()
x= pg.run_session(ids[1])
f = ps.plot_fit(ids[20]) 

# run all sessions
manifest = ps.get_manifest()
manifest = pg.run_manifest(manifest)


manifest.to_pickle('/home/alex.piet/Desktop/manifest_with_ridge.pkl')

r2_vip = np.concatenate(manifest[manifest['cre_line'] == manifest.iloc[0]['cre_line']]['model_r2'].values)
r2_slc = np.concatenate(manifest[manifest['cre_line'] == manifest.iloc[-1]['cre_line']]['model_r2'].values)
plt.figure()
plt.hist(r2_slc,50)
plt.figure()
plt.hist(r2_vip,50)


coefs = manifest['model_coefs'].values
dummy = []
for i in range(0,len(coefs)):
    if len(coefs[i]) > 0:
        dummy.append(np.vstack(coefs[i]))
all_coefs = np.vstack(dummy)
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(all_coefs)
X = pca.transform(all_coefs)

# for each session, compute average response window around changes, hits, pref stim
plt.figure();
timestamps = session.trial_response_df.iloc[0].dff_trace_timestamps-session.trial_response_df.iloc[0].change_time
session_dff = np.mean(np.vstack(session.trial_response_df[(session.trial_response_df['go']) & (session.trial_response_df['hit'])]['dff_trace']),0)
plt.plot(timestamps,session_dff)

dffs = []
dexes = []
slc = []
for id in ps.get_session_ids():
    print(id)
    session = ps.get_data(id)
    good_dff = False
    good_fit = False
    if len(session.trial_response_df[(session.trial_response_df['go']) & (session.trial_response_df['hit'])]['dff_trace']) > 0:
        session_dff = np.mean(np.vstack(session.trial_response_df[(session.trial_response_df['go']) & (session.trial_response_df['hit'])]['dff_trace']),0)
        good_dff = True
    try:
        fit = ps.load_fit(id)
        dropout = np.empty((len(fit['models']),))
        for i in range(0,len(fit['models'])):
            dropout[i] = (1-fit['models'][i][1]/fit['models'][0][1])*100
        dex = -(dropout[2] - dropout[18])
        good_fit = True
    except:
        pass
    if good_dff & good_fit:
        dffs.append(session_dff)
        dexes.append(dex)
        slc.append(session.metadata['full_genotype'][0:3] == 'Slc')
        
all_dffs = np.vstack(dffs)

# Sort based on dexes
sort_dex = np.argsort(dexes)
sort_dexes = np.array(dexes)[sort_dex]
sort_dffs = all_dffs[sort_dex,:]
plt.figure()
plt.imshow(sort_dffs[np.array(slc),:])
slc_sorted = sort_dffs[np.array(slc),:]
vip_sorted = sort_dffs[~np.array(slc),:]

plt.figure()
slc_dexes = sort_dexes[slc]
slc_dexes[slc_dexes<-15] = -15
slc_dexes[slc_dexes> 15] = 15
slc_dexes = (slc_dexes -np.min(slc_dexes))/(np.max(slc_dexes)-np.min(slc_dexes))
for i in range(0,np.shape(slc_sorted)[0]):
    plt.plot(slc_sorted[i,:]/np.max(slc_sorted[i,:]),'-',color=xx(slc_dexes[i]),alpha=0.5)

plt.figure()
vip_dexes = sort_dexes[~np.array(slc)]
vip_dexes[vip_dexes<-15] = -15
vip_dexes[vip_dexes> 15] = 15
vip_dexes = (vip_dexes -np.min(vip_dexes))/(np.max(vip_dexes)-np.min(vip_dexes))
for i in range(0,np.shape(vip_sorted)[0]):
    plt.plot(vip_sorted[i,:],'-',color=xx(vip_dexes[i]),alpha=0.5)

plt.figure()
plt.imshow(sort_dffs[~np.array(slc),:])

id = ids[20]
session = ps.get_data(id)
fit = ps.load_fit(id)
pm.get_metrics(session)
cells = session.flash_response_df['cell_specimen_id'].unique()

all_vals = []
r2_vals =[]
for cell in cells:
    cell_flash_df = pg.get_cell_flash_df(cell,session,fit)
    siggy = (np.sum(cell_flash_df['p_value'] < 0.05)/len(cell_flash_df)) > 0.25
    if siggy:
        try:
            task = cell_flash_df[(cell_flash_df['pref_stim'])&(cell_flash_df['change'])]['task0'].values
            vals = cell_flash_df[(cell_flash_df['pref_stim'])&(cell_flash_df['change'])]['mean_response'].values
            reg =linear_model.LinearRegression()
            reg.fit(task.reshape(-1,1),vals)
            all_vals.append(reg.coef_[0])
            r2_vals.append(reg.score(task.reshape(-1,1),vals))
            #if r2_vals[-1] > 0.05:
            #    plt.figure()
            #    plt.plot(task,vals,'ko')
        except:
            pass

plt.figure()
plt.hist(np.array(all_vals),50)
plt.hist(np.array(all_vals)[np.array(r2_vals)> 0.05],50)

# plot distribution of slopes for each cell
# group cells by preferred stimulus
id = ids[21]
active_slc = np.intersect1d(ps.get_slc_session_ids(),ps.get_active_ids())
slc_375 = manifest[manifest['ophys_experiment_id'].isin(active_slc) & (manifest['imaging_depth']==375)]
slc_175 = manifest[manifest['ophys_experiment_id'].isin(active_slc) & (manifest['imaging_depth']==175)]
r2,coefs = get_list(slc_375['ophys_experiment_id'].values)
r2,coefs = get_list(slc_175['ophys_experiment_id'].values)

plt.figure()
plt.hist(coefs)


def get_list(ids):
    r2 = []
    coefs =[]
    for id in ids:
        print(id)
        try:
            a,b = pg.plot_summary(id)
            r2.append(b)
            coefs.append(a[0])
        except:
            pass
    return r2,coefs




# Design Notes
# not worrying about "in-bout" traces
# not worrying about cross-validated weights
# This version of the model doesnt have omissions0, but VIP cells probably prefer omissions!!

# Clean up code
# visualize errors/prediction?
# cre line differences?
# timing/task differences?


cms = []
dexes = []
active_slc = np.intersect1d(ps.get_slc_session_ids(),ps.get_active_ids())
stage = []
for id in active_slc:
    print(id)
    try:
        session = ps.get_data(id)
        cm,acm,f,t,b = pg.get_session_change_modulation(id) 
        fit = ps.load_fit(id)
        dropout = np.empty((len(fit['models']),))
        for i in range(0,len(fit['models'])):
            dropout[i] = (1-fit['models'][i][1]/fit['models'][0][1])*100
        dex = -(dropout[2] - dropout[18])
        cms.append(np.nanmean(np.hstack(cm)))      
        dexes.append(dex) 
        stage.append(session.metadata['stage'])
    except:
        pass

 
plt.figure()
plt.plot(cms,dexes,'ko')

cms = np.array(cms)
dexes = np.array(dexes)
d1 = [x[6] == '1' for x in stage]
d3 = [x[6] == '3' for x in stage]
d4 = [x[6] == '4' for x in stage]
d6 = [x[6] == '6' for x in stage]
plt.figure()
plt.plot(cms[d1],dexes[d1],'ko')
plt.plot(cms[d3],dexes[d3],'ro')
plt.plot(cms[d4],dexes[d4],'go')
plt.plot(cms[d6],dexes[d6],'bo')


cms = []
dexes = []
stages = []
active_slc = np.intersect1d(ps.get_slc_session_ids(),ps.get_active_ids())
for id in active_slc:
    print(id)
    try:
        cm,dex,stage = pg.streamline_get_session_change_modulation(id) 
        cms.append(np.nanmean(np.hstack(cm)))      
        dexes.append(dex) 
        stages.append(stage)
    except:
        pass


manifest = ps.get_manifest()
active_slc = np.intersect1d(ps.get_slc_session_ids(),ps.get_active_ids())
container = manifest[manifest['ophys_experiment_id'] == active_slc[1]]['container_id'].values[0]
manifest[manifest['container_id'] == container][['ophys_experiment_id','container_id','passive_session','stage_name']]
passive_matched = 797255551
novel_matched = 795076128
cm1,acm1,f1,t1,b1,hits1 = pg.streamline_get_session_change_modulation(active_slc[1])
cm2,acm2,f2,t2,b2,hits2 = pg.get_session_change_modulation(passive_matched)
cm3,acm3,f3,t3,b3,hits3 = pg.get_session_change_modulation(novel_matched)

cm1= pg.streamline_get_session_change_modulation(active_slc[1])
cm2= pg.streamline_get_session_change_modulation(passive_matched)
cm3= pg.streamline_get_session_change_modulation(novel_matched)
make_figure(cm1)
make_bar(cm1,cm2)

def make_figure2(cm1,cm2,nbins=75):
    plt.figure(figsize=(5,5))
    a_cm1 = np.array(np.hstack(cm1))
    a_cm2 = np.array(np.hstack(cm2))
    a_cm1 = a_cm1[(a_cm1 < 2)&(a_cm1>-2)]
    a_cm2 = a_cm2[(a_cm2 < 2)&(a_cm2>-2)]
    hist1,bin_edges = np.histogram(a_cm1,nbins)
    hist2,bin_edges = np.histogram(a_cm2,bin_edges)
    plt.bar(bin_edges[:-1],hist1/np.sum(hist1),width=np.diff(bin_edges)[0],color='gray',alpha=0.5)
    plt.bar(bin_edges[:-1],hist2/np.sum(hist2),width=np.diff(bin_edges)[0],color='gray',alpha=0.5)
    #plt.hist(a_cm1[(a_cm1 < 2)&(a_cm1>-2)],75,color='gray',alpha=0.5)
    #plt.hist(a_cm2[(a_cm2 < 2)&(a_cm2>-2)],75,color='blue',alpha=0.5)
    plt.ylabel('Cell-Trial Pairs',fontsize=16)
    plt.xlabel('Change Modulation Index',fontsize=16)
    plt.gca().axvline(0,linestyle='--',color='k',alpha=0.5)
    plt.gca().set_xticks([-1,0,1])
    plt.gca().set_xticklabels(['-1','0','1'],fontsize=14)
    plt.gca().tick_params(axis='both',labelsize=14)
    plt.tight_layout()

def make_bar(cm1,cm2):
    a_cm1 = np.array(np.hstack(cm1))
    a_cm2 = np.array(np.hstack(cm2))
    plt.figure(figsize=(4/1.5,5/1.5))
    mean_1 = np.mean(a_cm1)
    mean_2 = np.mean(a_cm2)
    sem1 = np.std(a_cm1)/np.sqrt(len(a_cm1))
    sem2 = np.std(a_cm2)/np.sqrt(len(a_cm2))
    plt.plot([0,1],[mean_1,mean_1],'k-',linewidth=5)
    plt.plot([0.5, 0.5], [mean_1 + sem1, mean_1-sem1], 'k-')    
    plt.plot([1,2],[mean_2,mean_2],'b-',linewidth=5)
    plt.plot([1.5, 1.5], [mean_2 + sem2, mean_2-sem2], 'b-')   
    plt.gca().tick_params(axis='both',labelsize=14)
    plt.ylabel('Change Modulation',fontsize=16)
    plt.gca().set_xticks([0.5,1.5])
    plt.gca().set_xticklabels(['Active','Passive'],fontsize=14)
    plt.gca().axhline(0,linestyle='--',color='k',alpha=0.5)
    plt.tight_layout()

def make_figure(cm1):
    plt.figure(figsize=(6.4/1.5,5.3/1.5))
    a_cm1 = np.array(np.hstack(cm1))
    a_cm1 = a_cm1[(a_cm1 < 2)&(a_cm1>-2)]
    plt.hist(a_cm1[(a_cm1 < 2)&(a_cm1>-2)],75,color='gray')
    plt.ylabel('Cell-Trial Pairs',fontsize=16)
    plt.xlabel('Change Modulation Index',fontsize=16)
    plt.gca().axvline(0,linestyle='--',color='k',alpha=0.5)
    plt.gca().set_xticks([-1,0,1])
    plt.gca().set_xticklabels(['-1','0','1'],fontsize=14)
    plt.gca().tick_params(axis='both',labelsize=14)
    plt.tight_layout()




