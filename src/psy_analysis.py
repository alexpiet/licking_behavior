import numpy as np
import pandas as pd
import psy_tools as ps
import psy_style as pstyle
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import ttest_ind

def plot_all_RT_by_engagement(ophys, version):
    RT_by_engagement(ophys,version,title='All Mice')
    RT_by_engagement(ophys.query('visual_strategy_session'),version,title='Visual')
    RT_by_engagement(ophys.query('not visual_strategy_session'),version,title='Timing')
    RT_by_engagement(ophys.query('visual_strategy_session'),version,title='Visual. Change', change_only=True)
    RT_by_engagement(ophys.query('not visual_strategy_session'),version,title='Timing. Change', change_only=True)

def plot_all_RT_by_group(ophys,version):
    RT_by_group(ophys,version,title='all_images_strategy_engaged')
    RT_by_group(ophys,version,title='all_images_strategy_disengaged',engaged=False)
    RT_by_group(ophys,version,title='change_images_strategy_engaged',change_only=True)
    RT_by_group(ophys,version,title='change_images_strategy_disengaged',engaged=False,change_only=True)
    RT_by_group(ophys.query('strategy_matched'),version, groups=['cre_line =="Slc17a7-IRES2-Cre"',
        'cre_line == "Vip-IRES-Cre"', 'cre_line =="Sst-IRES-Cre"'], labels=['Slc','Vip','Sst'],
        title='Cre line, Engaged', engaged=True,additional_label='strategy_matched')
    RT_by_group(ophys.query('strategy_matched'),version, groups=['cre_line =="Slc17a7-IRES2-Cre"',
        'cre_line == "Vip-IRES-Cre"', 'cre_line =="Sst-IRES-Cre"'], labels=['Slc','Vip','Sst'],
        title='Cre line, Disengaged', engaged=False,additional_label='strategy_matched')
    RT_by_group(ophys.query('strategy_matched'),version, groups=['cre_line =="Slc17a7-IRES2-Cre"',
        'cre_line == "Vip-IRES-Cre"', 'cre_line =="Sst-IRES-Cre"'], labels=['Slc','Vip','Sst'],
        title='Cre line, Change Images, Engaged', engaged=True,change_only=True, additional_label='strategy_matched')
    RT_by_group(ophys.query('strategy_matched'),version, groups=['cre_line =="Slc17a7-IRES2-Cre"',
        'cre_line == "Vip-IRES-Cre"', 'cre_line =="Sst-IRES-Cre"'], labels=['Slc','Vip','Sst'],
        title='Cre line, Change Images, Disengaged', engaged=False,change_only=True, additional_label='strategy_matched')
    RT_by_group(ophys, version, groups=['session_number == 1','session_number==3','session_number==4','session_number==6'],
        labels=['F1','F3','N1','N3'], engaged=True, title='By Session, Engaged')
    RT_by_group(ophys, version, groups=['session_number == 1','session_number==3','session_number==4','session_number==6'],
        labels=['F1','F3','N1','N3'], engaged=False, title='By Session, Disengaged')
    RT_by_group(ophys, version, groups=['session_number == 1','session_number==3','session_number==4','session_number==6'],
        labels=['F1','F3','N1','N3'], engaged=True, title='By Session, Engaged, Change Images',change_only=True)
    RT_by_group(ophys, version, groups=['session_number == 1','session_number==3','session_number==4','session_number==6'],
        labels=['F1','F3','N1','N3'], engaged=False, title='By Session, Disengaged, Change Images',change_only=True)
    RT_by_group(ophys.query('include_for_novel'), version, groups=['session_number==3','session_number==4'],
        labels=['F3','N1'], engaged=True, title='True Novel',additional_label='true_novel')  

def RT_by_group(ophys,version=None,bins=44,title='all',
    groups=['visual_strategy_session','not visual_strategy_session'],
    engaged=True,labels=['Visual Engaged','Timing Engaged'],change_only=False,density=True,
    additional_label=''
    ):

    plt.figure()
    colors= plt.get_cmap('tab10')
    #colors=['mediumblue','firebrick']
    for gindex, group in enumerate(groups):
        RT = []
        for index, row in ophys.query(groups[gindex]).iterrows():
            if engaged:
                vec = row['engaged']
                vec[np.isnan(vec)] = False
                vec = vec.astype(bool)
                if change_only:
                    c_vec = row['change']
                    c_vec[np.isnan(c_vec)]=False
                    vec = vec & c_vec.astype(bool)
            else:
                vec = row['engaged']
                vec[np.isnan(vec)] = True
                vec = ~vec.astype(bool)
                if change_only:
                    c_vec = row['change']
                    c_vec[np.isnan(c_vec)]=False
                    vec = vec & c_vec.astype(bool)
            RT.append(row['RT'][vec]) 
        RT = np.hstack(RT)
        plt.hist(RT, color=colors(gindex),alpha=1/len(groups),label=labels[gindex],bins=bins,density=density)

    plt.ylabel('Density',fontsize=16)
    plt.xlabel('RT (s)',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0,.75)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    filename = '_'.join(labels).lower().replace(' ','_')
    if engaged:
        filename += '_engaged'
    if change_only:
        filename += '_change_images'
    else:
        filename += '_all_images'
    filename += additional_label 
    directory = ps.get_directory(version)
    plt.savefig(directory+"figures_summary/summary_"+filename+"_RT_by_group.png")
    plt.savefig(directory+"figures_summary/summary_"+filename+"_RT_by_group.svg")

def RT_by_engagement(ophys,version=None,bins=44,title='all',change_only=False):
    engaged_color='k'
    disengaged_color='r'   
 
    # Aggregate data
    RT_engaged = []
    for index, row in ophys.iterrows():
        vec = row['engaged']
        vec[np.isnan(vec)] = False
        vec = vec.astype(bool)
        if change_only:
            c_vec = row['change']
            c_vec[np.isnan(c_vec)]=False
            vec = vec & c_vec.astype(bool)
        RT_engaged.append(row['RT'][vec])
    RT_disengaged = []
    for index, row in ophys.iterrows():
        vec = row['engaged']
        vec[np.isnan(vec)] = True
        vec = ~vec.astype(bool)
        if change_only:
            c_vec = row['change']
            c_vec[np.isnan(c_vec)]=False
            vec = vec & c_vec.astype(bool)
        RT_disengaged.append(row['RT'][vec]) 
    RT_engaged = np.hstack(RT_engaged)
    RT_disengaged = np.hstack(RT_disengaged)

    # Plot
    plt.figure()
    plt.hist(RT_engaged, color=engaged_color,alpha=.5,label='Engaged',bins=bins)
    plt.hist(RT_disengaged, color=disengaged_color,alpha=.5,label='Disengaged',bins=bins)
    plt.ylabel('count',fontsize=16)
    plt.xlabel('RT (s)',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0,.75)
    plt.legend()
    plt.title(title)
    plt.tight_layout()

    # Save
    directory = ps.get_directory(version)
    plt.savefig(directory+"figures_summary/summary_"+title.lower().replace(' ','_')+"_RT_by_engagement.png")
    plt.savefig(directory+"figures_summary/summary_"+title.lower().replace(' ','_')+"_RT_by_engagement.svg")









# UPDATE_REQUIRED
def pivot_manifest_by_stage(manifest,key='strategy_dropout_index',mean_subtract=True):

    x = manifest[['specimen_id','session_number',key]]
    x_pivot = pd.pivot_table(x,values=key,index='specimen_id',columns=['session_number'])
    x_pivot['mean_index'] = [np.nanmean(x) for x in zip(x_pivot[1],x_pivot[3],x_pivot[4],x_pivot[6])]

    if mean_subtract:
        x_pivot['mean_1'] = x_pivot[1] - x_pivot['mean_index']
        x_pivot['mean_3'] = x_pivot[3] - x_pivot['mean_index']
        x_pivot['mean_4'] = x_pivot[4] - x_pivot['mean_index']
        x_pivot['mean_6'] = x_pivot[6] - x_pivot['mean_index']
    else:
        x_pivot['mean_1'] = x_pivot[1]
        x_pivot['mean_3'] = x_pivot[3]
        x_pivot['mean_4'] = x_pivot[4]
        x_pivot['mean_6'] = x_pivot[6]
    return x_pivot

# UPDATE_REQUIRED
def plot_pivoted_manifest_by_stage(manifest, key='strategy_dropout_index',w=.45,flip_index=False,version=None,savefig=True,label=None,mean_subtract=True):
    if flip_index:
        manifest = manifest.copy()
        manifest[key] = -manifest[key]
    x_pivot = pivot_manifest_by_stage(manifest, key=key,mean_subtract=mean_subtract)
    plt.figure(figsize=(5,5))
    stages = [1,3,4,6]
    counts = [1,2,3,4]
    colors = pstyle.get_project_colors()
    mapper = {
        1:'F1',
        3:'F3',
        4:'N1',
        6:'N3'}
    for val in zip(counts,stages):
        m = x_pivot['mean_'+str(val[1])].mean()
        s = x_pivot['mean_'+str(val[1])].std()/np.sqrt(len(x_pivot))
        plt.plot([val[0]-w,val[0]+w],[m,m],linewidth=4,color=colors[mapper[val[1]]])
        plt.plot([val[0],val[0]],[m+s,m-s],linewidth=1,color='gray')
    pval = ttest_ind(x_pivot[3].values, x_pivot[4].values,nan_policy='omit')
    ylim = plt.ylim()[1]
    r = plt.ylim()[1] - plt.ylim()[0]
    sf = .075
    offset = 2 
    plt.plot([2,3],[ylim+r*sf, ylim+r*sf],'k-')
    plt.plot([2,2],[ylim, ylim+r*sf], 'k-')
    plt.plot([3,3],[ylim, ylim+r*sf], 'k-')
    
    if pval[1] < 0.05:
        plt.plot(2.5, ylim+r*sf*1.5,'k*')
    else:
        plt.text(2.5,ylim+r*sf*1.25, 'ns')


    if label is None:
        label = key
    plt.ylabel('$\Delta$ '+label,fontsize=24)
    plt.xlabel('Session #',fontsize=24)
    plt.yticks(fontsize=16)
    plt.xticks(counts,['F1','F3','N1','N3'],fontsize=24)
    plt.gca().axhline(0,color='k',linestyle='--',alpha=.25)
    plt.tight_layout()
    directory = ps.get_directory(version)  
    if savefig:
        plt.savefig(directory+'figures_summary/relative_by_stage_'+key+'.svg')
        plt.savefig(directory+'figures_summary/relative_by_stage_'+key+'.png')
 

def plot_all_pivoted(manifest, version,force_novel=True):
    if force_novel:
        manifest = manifest.query('include_for_novel').copy()
    plot_pivoted_manifest_by_stage(manifest, key='dropout_task0', flip_index=True, version=version, label='Visual Index')
    plot_pivoted_manifest_by_stage(manifest, key='dropout_timing1D', flip_index=True, version=version, label='Timing Index')
    plot_pivoted_manifest_by_stage(manifest, key='dropout_omissions1', flip_index=True, version=version, label='Prev. Omission Index')
    plot_pivoted_manifest_by_stage(manifest, key='dropout_omissions', flip_index=True, version=version, label='Omission Index')
    plot_pivoted_manifest_by_stage(manifest, key='strategy_dropout_index', version=version, label='Dropout Index')
    plot_pivoted_manifest_by_stage(manifest, key='strategy_weight_index', version=version, label='Weight Index')
    plot_pivoted_manifest_by_stage(manifest, key='lick_hit_fraction', version=version, label='Lick Hit Fraction')
    plot_pivoted_manifest_by_stage(manifest, key='lick_fraction', version=version, label='Lick Fraction')
    plot_pivoted_manifest_by_stage(manifest, key='num_hits', version=version, label='Hits/Session')
    plot_pivoted_manifest_by_stage(manifest, key='strategy_weight_index_1st_half', version=version, label='Weight Index (1st Half)')
    plot_pivoted_manifest_by_stage(manifest, key='strategy_weight_index_2nd_half', version=version, label='Weight Index (2nd Half)')
    plot_pivoted_manifest_by_stage(manifest, key='visual_weight_index_engaged', version=version, label='visual_weight_index_engaged')
    plot_pivoted_manifest_by_stage(manifest, key='visual_weight_index_disengaged', version=version, label='visual_weight_index_disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='timing_weight_index_engaged', version=version, label='timing_weight_index_engaged')
    plot_pivoted_manifest_by_stage(manifest, key='timing_weight_index_disengaged', version=version, label='timing_weight_index_disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='omissions_weight_index_engaged', version=version, label='omissions_weight_index_engaged')
    plot_pivoted_manifest_by_stage(manifest, key='omissions_weight_index_disengaged', version=version, label='omissions_weight_index_disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='omissions1_weight_index_engaged', version=version, label='omissions1_weight_index_engaged')
    plot_pivoted_manifest_by_stage(manifest, key='omissions1_weight_index_disengaged', version=version, label='omissions1_weight_index_disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='bias_weight_index_engaged', version=version, label='bias_weight_index_engaged')
    plot_pivoted_manifest_by_stage(manifest, key='bias_weight_index_disengaged', version=version, label='bias_weight_index_disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='strategy_weight_index_engaged', version=version, label='strategy_weight_index_engaged')
    plot_pivoted_manifest_by_stage(manifest, key='strategy_weight_index_disengaged', version=version, label='strategy_weight_index_disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='lick_hit_fraction_rate_engaged', version=version, label='lick_hit_fraction_rate_engaged')
    plot_pivoted_manifest_by_stage(manifest, key='lick_hit_fraction_rate_disengaged', version=version, label='lick_hit_fraction_rate_disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='hit_engaged', version=version, label='hit_engaged')
    plot_pivoted_manifest_by_stage(manifest, key='hit_disengaged', version=version, label='hit_disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='CR_engaged', version=version, label='correct rejects engaged')
    plot_pivoted_manifest_by_stage(manifest, key='CR_disengaged', version=version, label='correct rejects disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='miss_engaged', version=version, label='misses engaged')
    plot_pivoted_manifest_by_stage(manifest, key='miss_disengaged', version=version, label='misses disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='lick_bout_rate_engaged', version=version, label='lick_bout_rate engaged')
    plot_pivoted_manifest_by_stage(manifest, key='lick_bout_rate_disengaged', version=version, label='lick_bout_rate disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='FA_engaged', version=version, label='False Alarms Engaged')
    plot_pivoted_manifest_by_stage(manifest, key='FA_disengaged', version=version, label='False Alarms Disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='reward_rate_engaged', version=version, label='Reward Rate Engaged')
    plot_pivoted_manifest_by_stage(manifest, key='reward_rate_disengaged', version=version, label='Reward Rate Disengaged')
    plot_pivoted_manifest_by_stage(manifest, key='RT_engaged', version=version, label='RT Engaged')
    plot_pivoted_manifest_by_stage(manifest, key='RT_disengaged', version=version, label='RT Disengaged')


#### Dev below here
def triggered_analysis(ophys, version=None,triggers=['hit','FA'],dur=50,responses=['lick_hit_fraction','lick_bout_rate']):
    # Iterate over sessions

    plt.figure()
    for trigger in triggers:
        for response in responses:
            stas =[]
            for index, row in tqdm(ophys.iterrows(),total=ophys.shape[0]):
                stas.append(session_triggered_analysis(row, trigger, response,dur))
            plt.plot(np.nanmean(stas,0),label=response+' by '+trigger)

    plt.legend()

def session_triggered_analysis(ophys_row,trigger,response, dur):
    indexes = np.where(ophys_row[trigger] ==1)[0]
    vals = []
    for index in indexes:
        vals.append(get_aligned(ophys_row[response],index, length=dur))
    if len(vals) >1:
        mean= np.mean(np.vstack(vals),0)
        mean = mean - mean[0]
    else:
        mean = np.array([np.nan]*dur)
    return mean

def plot_triggered_analysis(ophys_row,trigger,responses,dur):
    plt.figure()
    for response in responses:
        sta = session_triggered_analysis(ophys_row,trigger, response,dur)
        plt.plot(sta, label=response)
        plt.plot(sta+sem1,'k')
        plt.plot(sta-sem1,'k')       
   
    plt.axhline(0,color='k',linestyle='--',alpha=.5) 
    plt.ylabel('change relative to hit/FA')
    plt.xlabel(' flash #') 
    plt.legend()

def get_aligned(vector, start, length=4800):

    if len(vector) >= start+length:
        aligned= vector[start:start+length]
    else:
        aligned = np.concatenate([vector[start:], [np.nan]*(start+length-len(vector))])
    return aligned





