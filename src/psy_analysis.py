import numpy as np
import pandas as pd
import psy_style as pstyle
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import psy_general_tools as pgt


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
    directory = pgt.get_directory(version)  
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


## Event Triggered Analysis
#######################################################################
def triggered_analysis(ophys, version=None,triggers=['hit','miss'],dur=50,responses=['lick_bout_rate']):
    # Iterate over sessions

    plt.figure()
    for trigger in triggers:
        for response in responses:
            stas =[]
            skipped = 0
            for index, row in ophys.iterrows():
                try:
                    stas.append(session_triggered_analysis(row, trigger, response,dur))
                except:
                    pass
            mean = np.nanmean(stas,0)
            n=np.shape(stas)[0]
            std = np.nanstd(stas,0)/np.sqrt(n)

            plt.plot(mean,label=response+' by '+trigger)
            plt.plot(mean+std,'k')
            plt.plot(mean-std,'k')       
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

def plot_triggered_analysis(row,trigger,responses,dur):
    plt.figure()
    for response in responses:
        sta = session_triggered_analysis(row,trigger, response,dur)
        plt.plot(sta, label=response)
        #plt.plot(sta+sem1,'k')
        #plt.plot(sta-sem1,'k')       
   
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





