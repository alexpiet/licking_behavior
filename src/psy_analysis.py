import numpy as np
import pandas as pd
import psy_style as pstyle
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import psy_general_tools as pgt


def pivot_df_by_stage(summary_df,key='strategy_dropout_index',mean_subtract=True):
    '''
        doc string # TODO
    '''
    # TODO
    # Should this operate on session_number?
    # does it handle out of order sessions?
    # is there a better way to do the mean subtraction?
        # Should I really make the columns mean_ instead of just 1, 3, 4, 6?
    # Validate pivot computation
    x = summary_df[['mouse_id','session_number',key]]
    x_pivot = pd.pivot_table(x,values=key,index='mouse_id',columns=['session_number'])
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

def plot_pivoted_df_by_stage(summary_df, key,version,flip_index=False,mean_subtract=True,savefig=False,group=None):
    '''
        doc string #TODO
    '''
    # Get pivoted data
    if flip_index:
        summary_df = summary_df.copy()
        summary_df[key] = -summary_df[key]
    x_pivot = pivot_df_by_stage(summary_df, key=key,mean_subtract=mean_subtract)

    # Set up Figure
    fig, ax = plt.subplots()
    colors = pstyle.get_project_colors()
    style = pstyle.get_style()
    stages = [1,3,4,6]
    mapper = {1:'F1',3:'F3',4:'N1',6:'N3'}
    w=.45,

    # Plot each stage
    for index,val in enumerate(stages):
        m = x_pivot['mean_'+str(val)].mean()
        s = x_pivot['mean_'+str(val)].std()/np.sqrt(len(x_pivot))
        plt.plot([index-w,index+w],[m,m],linewidth=4,color=colors[mapper[val]])
        plt.plot([index,index],[m+s,m-s],linewidth=1,color=colors[mapper[val]])
    
    # Add Statistics
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

    # Clean up Figure
    label = pgt.get_clean_string([key])[0]
    plt.ylabel('$\Delta$ '+label,fontsize=style['label_fontsize'])
    plt.xlabel('Session #',fontsize=style['label_fontsize'])
    plt.yticks(fontsize=style['axis_ticks_fontsize'])
    plt.xticks(range(0,len(stages)),[mapper[x] for x in stages],
        fontsize=style['axis_ticks_fontsize'])
    ax.axhline(0,color=style['axline_color'],linestyle=style['axline_linestyle'],
        alpha=style['axline_alpha'])
    plt.tight_layout()

    # Save Figure
    if savefig:
        directory = pgt.get_directory(version,subdirectory='figures',group=group)  
        filename = directory+'relative_by_stage_'+key+'.png'
        print('Figure saved to: '+filename)
        plt.savefig(filename)
 

def plot_all_pivoted(summary_df, version,force_novel=True):
    if force_novel:
        summary_df = summary_df.query('include_for_novel').copy()
    plot_pivoted_df_by_stage(summary_df, key='dropout_task0', flip_index=True, version=version, label='Visual Index')
    plot_pivoted_df_by_stage(summary_df, key='dropout_timing1D', flip_index=True, version=version, label='Timing Index')
    plot_pivoted_df_by_stage(summary_df, key='dropout_omissions1', flip_index=True, version=version, label='Prev. Omission Index')
    plot_pivoted_df_by_stage(summary_df, key='dropout_omissions', flip_index=True, version=version, label='Omission Index')
    plot_pivoted_df_by_stage(summary_df, key='strategy_dropout_index', version=version, label='Dropout Index')
    plot_pivoted_df_by_stage(summary_df, key='strategy_weight_index', version=version, label='Weight Index')
    plot_pivoted_df_by_stage(summary_df, key='lick_hit_fraction', version=version, label='Lick Hit Fraction')
    plot_pivoted_df_by_stage(summary_df, key='lick_fraction', version=version, label='Lick Fraction')
    plot_pivoted_df_by_stage(summary_df, key='num_hits', version=version, label='Hits/Session')
    plot_pivoted_df_by_stage(summary_df, key='visual_weight_index_engaged', version=version, label='visual_weight_index_engaged')
    plot_pivoted_df_by_stage(summary_df, key='visual_weight_index_disengaged', version=version, label='visual_weight_index_disengaged')
    plot_pivoted_df_by_stage(summary_df, key='timing_weight_index_engaged', version=version, label='timing_weight_index_engaged')
    plot_pivoted_df_by_stage(summary_df, key='timing_weight_index_disengaged', version=version, label='timing_weight_index_disengaged')
    plot_pivoted_df_by_stage(summary_df, key='omissions_weight_index_engaged', version=version, label='omissions_weight_index_engaged')
    plot_pivoted_df_by_stage(summary_df, key='omissions_weight_index_disengaged', version=version, label='omissions_weight_index_disengaged')
    plot_pivoted_df_by_stage(summary_df, key='omissions1_weight_index_engaged', version=version, label='omissions1_weight_index_engaged')
    plot_pivoted_df_by_stage(summary_df, key='omissions1_weight_index_disengaged', version=version, label='omissions1_weight_index_disengaged')
    plot_pivoted_df_by_stage(summary_df, key='bias_weight_index_engaged', version=version, label='bias_weight_index_engaged')
    plot_pivoted_df_by_stage(summary_df, key='bias_weight_index_disengaged', version=version, label='bias_weight_index_disengaged')
    plot_pivoted_df_by_stage(summary_df, key='strategy_weight_index_engaged', version=version, label='strategy_weight_index_engaged')
    plot_pivoted_df_by_stage(summary_df, key='strategy_weight_index_disengaged', version=version, label='strategy_weight_index_disengaged')
    plot_pivoted_df_by_stage(summary_df, key='lick_hit_fraction_rate_engaged', version=version, label='lick_hit_fraction_rate_engaged')
    plot_pivoted_df_by_stage(summary_df, key='lick_hit_fraction_rate_disengaged', version=version, label='lick_hit_fraction_rate_disengaged')
    plot_pivoted_df_by_stage(summary_df, key='hit_engaged', version=version, label='hit_engaged')
    plot_pivoted_df_by_stage(summary_df, key='hit_disengaged', version=version, label='hit_disengaged')
    plot_pivoted_df_by_stage(summary_df, key='CR_engaged', version=version, label='correct rejects engaged')
    plot_pivoted_df_by_stage(summary_df, key='CR_disengaged', version=version, label='correct rejects disengaged')
    plot_pivoted_df_by_stage(summary_df, key='miss_engaged', version=version, label='misses engaged')
    plot_pivoted_df_by_stage(summary_df, key='miss_disengaged', version=version, label='misses disengaged')
    plot_pivoted_df_by_stage(summary_df, key='lick_bout_rate_engaged', version=version, label='lick_bout_rate engaged')
    plot_pivoted_df_by_stage(summary_df, key='lick_bout_rate_disengaged', version=version, label='lick_bout_rate disengaged')
    plot_pivoted_df_by_stage(summary_df, key='FA_engaged', version=version, label='False Alarms Engaged')
    plot_pivoted_df_by_stage(summary_df, key='FA_disengaged', version=version, label='False Alarms Disengaged')
    plot_pivoted_df_by_stage(summary_df, key='reward_rate_engaged', version=version, label='Reward Rate Engaged')
    plot_pivoted_df_by_stage(summary_df, key='reward_rate_disengaged', version=version, label='Reward Rate Disengaged')
    plot_pivoted_df_by_stage(summary_df, key='RT_engaged', version=version, label='RT Engaged')
    plot_pivoted_df_by_stage(summary_df, key='RT_disengaged', version=version, label='RT Disengaged')


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





