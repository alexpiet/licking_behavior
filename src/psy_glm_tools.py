import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
import seaborn as sns
import ternary
import random
from tqdm import tqdm
import hierarchical_boot as hb
import psy_general_tools as pgt

def get_cell_df(cell_id, session):
    return session.flash_response_df[session.flash_response_df['cell_specimen_id'] == cell_id]

def test_cell_reliability(cell_flash_df, pval=0.05, percent=0.25):
    return (np.sum(cell_flash_df['p_value'] < pval)/len(cell_flash_df)) > percent

def cell_change_modulation(cell, session):
    '''

    '''
    cell_flash_df = get_cell_df(cell,session)
    cell_flash_df = cell_flash_df[cell_flash_df['pref_stim']]

    cms = []
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],
        'change_modulation':[],'block_number':[],'change_modulation_base':[],'pref_stim':[], 
        'reliable_cell':[], 'good_response':[], 'dff_trace':[],'good_block':[],'good_response_base':[],
        'timestamps':[],'mean_response_0':[],'mean_response_9':[],'mean_response_0_base':[],'mean_response_9_base':[]})

    reliable_cell = test_cell_reliability(cell_flash_df,percent=0.25)  
    blocks = cell_flash_df['block_index'].unique()
    timestamps = session.ophys_timestamps
    cell_dff_trace = session.dff_traces.query('cell_specimen_id ==@cell').iloc[0]['dff']
    for block in blocks:        
        block_df = cell_flash_df[cell_flash_df['block_index'] == block] 
        if len(block_df) > 10:
            this_change= block_df.iloc[0]['mean_response']
            this_change_base = block_df.iloc[0]['baseline_response']
            this_non= block_df.iloc[9]['mean_response']
            this_non_base = block_df.iloc[9]['baseline_response']
            this_cm = (this_change - this_non)/(this_change+this_non)
            this_cm_base = ((this_change-this_change_base) - (this_non-this_non_base))/((this_change-this_change_base)+(this_non - this_non_base))
            dff_trace = cell_dff_trace[(timestamps > block_df.iloc[0]['start_time']) & (timestamps < block_df.iloc[9]['stop_time']+0.5)]
            ophys_timestamps = timestamps[(timestamps > block_df.iloc[0]['start_time']) & (timestamps < block_df.iloc[9]['stop_time']+0.5)]
        else:
            this_change= block_df.iloc[0]['mean_response']
            this_change_base = block_df.iloc[0]['baseline_response']
            this_non= block_df.iloc[-1]['mean_response']
            this_non_base = block_df.iloc[-1]['baseline_response']
            this_cm = (this_change - this_non)/(this_change+this_non)
            this_cm_base = ((this_change-this_change_base) - (this_non-this_non_base))/((this_change-this_change_base)+(this_non - this_non_base))
            dff_trace = cell_dff_trace[(timestamps > block_df.iloc[0]['start_time']) & (timestamps < block_df.iloc[-1]['stop_time']+0.5)]
            ophys_timestamps = timestamps[(timestamps > block_df.iloc[0]['start_time']) & (timestamps < block_df.iloc[-1]['stop_time']+0.5)]

        good_response = (this_cm > -1) & (this_cm < 1) &(this_cm_base > -1) & (this_cm_base < 1)
        good_response_base = (this_cm_base > -1) & (this_cm_base < 1)
        good_block = len(block_df) > 10
        
        if good_response & good_block:
            cms.append(this_cm)
        d = {'ophys_experiment_id':session.metadata['ophys_experiment_id'],'stage':session.metadata['stage'],
            'cell':cell,'imaging_depth':session.metadata['imaging_depth'],'change_modulation':this_cm,
            'block_number':block,'change_modulation_base':this_cm_base,'pref_stim':cell_flash_df.iloc[0]['image_name'],
            'reliable_cell':reliable_cell,'good_response':good_response,'dff_trace':dff_trace[0:232],'good_block':good_block,
            'good_response_base':good_response_base,'timestamps':ophys_timestamps[0:232],'mean_response_0':this_change, 'mean_response_9':this_non,
            'mean_response_0_base':this_change_base, 'mean_response_9_base':this_non_base}
        df = df.append(d,ignore_index=True)

    return df, cms

def session_change_modulation(id):
    session = pgt.get_data(id)
    cells = session.flash_response_df['cell_specimen_id'].unique()
    all_cms = []
    mean_cms = []
    var_cms = []
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],
        'change_modulation':[],'block_number':[],'change_modulation_base':[],'pref_stim':[], 
        'reliable_cell':[], 'good_response':[], 'dff_trace':[],'good_block':[],'good_response_base':[],
        'timestamps':[],'mean_response_0':[],'mean_response_9':[],'mean_response_0_base':[],'mean_response_9_base':[]})
    for cell in cells:
        cell_df,cm = cell_change_modulation(cell,session)
        if len(cm) > 0:
            all_cms.append(cm)
            mean_cms.append(np.mean(cm))
            var_cms.append(np.var(cm))
            df = df.append(cell_df,ignore_index=True)
    session_mean = np.mean(mean_cms)
    session_var = np.var(mean_cms)
    return df, all_cms,mean_cms,var_cms, session_mean, session_var

def manifest_change_modulation(ids,dc_offset=0.05):
    all_cms = []
    mean_cms = []
    var_cms = []
    session_means = []
    session_vars = []
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],
        'change_modulation':[],'block_number':[],'change_modulation_base':[],'pref_stim':[], 
        'reliable_cell':[], 'good_response':[], 'dff_trace':[],'good_block':[],'good_response_base':[],
        'timestamps':[],'mean_response_0':[],'mean_response_9':[],'mean_response_0_base':[],'mean_response_9_base':[]})
    for id in tqdm(ids):
        session_df, cell_cms,cell_mean_cms,cell_var_cms, session_mean,session_var = session_change_modulation(id)
        all_cms.append(cell_cms)
        mean_cms.append(cell_mean_cms)
        var_cms.append(cell_var_cms)
        session_means.append(session_mean)
        session_vars.append(session_var)
        df = df.append(session_df,ignore_index=True)

    df['good_block'] = df['good_block'] == 1.0
    df['good_response'] = df['good_response'] == 1.0
    df['good_response_base'] = df['good_response_base'] == 1.0
    df['reliable_cell'] = df['reliable_cell'] == 1.0

    df['change_modulation_dc'] = (df['mean_response_0']+dc_offset-(df['mean_response_9']+dc_offset))/(df['mean_response_0']+dc_offset+df['mean_response_9']+dc_offset) 
    df['good_response_dc'] = (df['change_modulation_dc'] > -1 ) & (df['change_modulation_dc'] < 1)

    df['change_modulation_base_dc'] = ((df['mean_response_0']-df['mean_response_0_base'])+dc_offset-((df['mean_response_9']-df['mean_response_9_base'])+dc_offset))/((df['mean_response_0']-df['mean_response_0_base'])+dc_offset+(df['mean_response_9']-df['mean_response_0'])+dc_offset) 
    df['good_response_base_dc'] = (df['change_modulation_base_dc'] > -1 ) & (df['change_modulation_base_dc'] < 1)

    df['real_response'] = (df['mean_response_0'] + df['mean_response_9']) > 0.1
    return df, all_cms, mean_cms, var_cms,session_means, session_vars, np.mean(session_means), np.var(session_means)

def plot_manifest_change_modulation_df(df,box_plot=True,plot_cells=True,metric='change_modulation',titlestr="",filepath=None):
    plt.figure()
    count = 0
    colors = sns.color_palette(n_colors=len(df['ophys_experiment_id'].unique()))
    this_session_means = df.groupby(['ophys_experiment_id','cell']).mean()[metric].groupby('ophys_experiment_id').mean().values
    session_ids = df.groupby(['ophys_experiment_id','cell']).mean()[metric].groupby('ophys_experiment_id').mean().index.values
    session_means_sorted = sorted(this_session_means)
    session_ids_sorted = [x for _,x in sorted(zip(this_session_means,session_ids))]

    for session_index, session_id in enumerate(session_ids_sorted):
        session_cell_means = df[df['ophys_experiment_id'] == session_id].groupby(['cell']).mean()[metric].values
        if plot_cells:
            # Sort Cells in this session
            session_cell_means_ids = df[df['ophys_experiment_id'] == session_id].groupby(['cell']).mean()[metric].index.values
            session_cell_ids_sorted = [x for _,x in sorted(zip(session_cell_means,session_cell_means_ids))]
            session_cell_means_sorted = sorted(session_cell_means)

            # Plot each cell/flash pair in this session 
            for index,this_cell in enumerate(session_cell_ids_sorted):
                this_cell_cms = df[(df['ophys_experiment_id'] == session_id) & (df['cell'] == this_cell)][metric].values
                if box_plot:
                    bplot = plt.gca().boxplot(this_cell_cms,showfliers=False,positions=[count],widths=1,patch_artist=True,showcaps=False)
                    for whisker in bplot['whiskers']:
                        whisker.set_color(colors[session_index])
                    bplot['medians'][0].set_color(colors[session_index])
                    for patch in bplot['boxes']:
                        patch.set_facecolor(colors[session_index])
                        patch.set_edgecolor(colors[session_index])
                else:
                    plt.plot(np.repeat(count,len(this_cell_cms)), this_cell_cms,'o',color=colors[session_index])
                plt.plot(count, session_cell_means_sorted[index],'ko',zorder=5) 
                count +=1

            # Plot session mean
            plt.plot([count-len(session_cell_ids_sorted),count-1],[session_means_sorted[session_index], session_means_sorted[session_index]],color=colors[session_index],label=str(session_index),linewidth=1,zorder=10)
            plt.plot([count-len(session_cell_ids_sorted),count-1],[session_means_sorted[session_index], session_means_sorted[session_index]],'k',linewidth=4,zorder=10)
        else:
            bplot= plt.gca().boxplot(session_cell_means,showfliers=False,positions=[session_index],widths=1,patch_artist=True,showcaps=False)
            for whisker in bplot['whiskers']:
                whisker.set_color(colors[session_index])
            bplot['medians'][0].set_color(colors[session_index])
            for patch in bplot['boxes']:
                patch.set_facecolor(colors[session_index])
                patch.set_edgecolor(colors[session_index])
            plt.plot([session_index-0.4,session_index+0.4],[session_means_sorted[session_index], session_means_sorted[session_index]],color=colors[session_index],label=str(session_index),linewidth=1,zorder=10)
            plt.plot([session_index-0.4,session_index+0.4],[session_means_sorted[session_index], session_means_sorted[session_index]],'k',linewidth=4,zorder=10)

    # clean up plot
    plt.xticks([])
    plt.ylim([-1,1])
    plt.gca().axhline(0,linestyle='--',color='k',alpha=1)
    plt.ylabel(metric)
    plt.xlabel('Cell')
    plt.title(titlestr)
    if plot_cells:
        plt.xlabel('Cell')
    else:
        plt.xlabel('Sessions')
    if type(filepath) is not type(None):
        plt.savefig(filepath+"_"+titlestr+".svg")


def plot_manifest_change_modulation(cell_cms, cell_mean_cms, session_means,box_plot=True,plot_cells=True,titlestr=""):
    plt.figure()
    count = 0
    colors = sns.color_palette(n_colors=len(session_means))
    cell_cms_sorted = [x for _,x in sorted(zip(session_means, cell_cms))]
    cell_mean_cms_sorted = [x for _,x in sorted(zip(session_means, cell_mean_cms))]
    session_means_sorted = sorted(session_means)

    for session_index, session_cell_cms in enumerate(cell_cms_sorted):
        if plot_cells:
            # Sort Cells in this session
            session_cell_cms_sorted = [x for _,x in sorted(zip(cell_mean_cms_sorted[session_index],session_cell_cms))]
            session_cell_mean_cms_sorted = sorted(cell_mean_cms_sorted[session_index])
            # Plot each cell/flash pair in this session 
            for index,this_cell_cms in enumerate(session_cell_cms_sorted):
                if box_plot:
                    bplot = plt.gca().boxplot(this_cell_cms,showfliers=False,positions=[count],widths=1,patch_artist=True,showcaps=False)
                    for whisker in bplot['whiskers']:
                        whisker.set_color(colors[session_index])
                    bplot['medians'][0].set_color(colors[session_index])
                    for patch in bplot['boxes']:
                        patch.set_facecolor(colors[session_index])
                        patch.set_edgecolor(colors[session_index])
                else:
                    plt.plot(np.repeat(count,len(this_cell_cms)), this_cell_cms,'o',color=colors[session_index])
                plt.plot(count, session_cell_mean_cms_sorted[index],'ko',zorder=5) 
                count +=1

            # Plot session mean
            plt.plot([count-len(session_cell_cms),count-1],[session_means_sorted[session_index], session_means_sorted[session_index]],color=colors[session_index],label=str(session_index),linewidth=1,zorder=10)
            plt.plot([count-len(session_cell_cms),count-1],[session_means_sorted[session_index], session_means_sorted[session_index]],'k',linewidth=4,zorder=10)
        else:
            bplot= plt.gca().boxplot(cell_mean_cms_sorted[session_index],showfliers=False,positions=[session_index],widths=1,patch_artist=True,showcaps=False)
            for whisker in bplot['whiskers']:
                whisker.set_color(colors[session_index])
            bplot['medians'][0].set_color(colors[session_index])
            for patch in bplot['boxes']:
                patch.set_facecolor(colors[session_index])
                patch.set_edgecolor(colors[session_index])
            plt.plot([session_index-0.4,session_index+0.4],[session_means_sorted[session_index], session_means_sorted[session_index]],color=colors[session_index],label=str(session_index),linewidth=1,zorder=10)
            plt.plot([session_index-0.4,session_index+0.4],[session_means_sorted[session_index], session_means_sorted[session_index]],'k',linewidth=4,zorder=10)

    # clean up plot
    plt.xticks([])
    plt.ylim([-1,1])
    plt.gca().axhline(0,linestyle='--',color='k',alpha=1)
    plt.ylabel('Change Modulation')
    plt.xlabel('Cell')
    plt.title(titlestr)
    if plot_cells:
        plt.xlabel('Cell')
    else:
        plt.xlabel('Sessions')


def plot_session_change_modulation(cell_cms, cell_mean_cms, session_mean,box_plot=True):
    plot_manifest_change_modulation([cell_cms],[cell_mean_cms],[session_mean],box_plot=box_plot)


def plot_simplex(points, labels,class_label,colors,norm_vars):
    figure, tax = ternary.figure(scale=1.0)
    figure.set_size_inches(5, 5)
    tax.boundary()
    tax.gridlines(multiple=0.2, color="black")
    tax.get_axes().axis('off')
    tax.clear_matplotlib_ticks()
    fontsize=18
    tax.right_corner_label(labels[0], fontsize=fontsize)
    tax.top_corner_label(labels[1], fontsize=fontsize)
    tax.left_corner_label(labels[2], fontsize=fontsize) 
    for index, p in enumerate(points):
        p = p/np.sum(p) 
        tax.plot([p], marker='o', color=colors[index], label=class_label[index], ms=norm_vars[index]*10,markeredgecolor='k',alpha=0.25)
    tax.legend(loc='upper right')
    tax.show()
    plt.tight_layout()

def bootstrap_session_cell_modulation(session_cms,numboots):
    all_session_cms = np.hstack(session_cms)
    num_cms_per_cell = [len(x) for x in session_cms]
    num_cells = len(session_cms)
    boot_cms=[]
    boot_mean_cms = []
    for i in range(0,numboots):
        this_cms = random.sample(list(all_session_cms),num_cms_per_cell[np.mod(i,num_cells)])
        boot_cms.append(this_cms)
        boot_mean_cms.append(np.mean(this_cms))
    return boot_cms, boot_mean_cms, np.mean(boot_mean_cms)

def shuffle(df, n=1, axis=0):     
    shuffle_df = df.copy()
    #shuffle_df.apply(lambda x: x.sample(frac=1).values)
    shuffle_df = shuffle_df.apply(np.random.permutation,axis=axis)
    return shuffle_df
    
def bootstrap_session_cell_modulation_df(df,numboots):
    shuffle_df = shuffle(df)
    for i in range(0,numboots):
        this_df = shuffle(df)
        this_df['cell'] = this_df['cell'] + 10000*(i+1)
        shuffle_df = shuffle_df.append(this_df,ignore_index=True)
    return shuffle_df

def compare_flash_dist_df(dfs,bins,colors,labels,alpha, ylabel="",xlabel="",metric='change_modulation',filepath=None):
    dists = []
    for index, df in enumerate(dfs):     
        dist = df[metric].values
        dists.append(dist)
    compare_dist(dists,bins,colors,labels,alpha, ylabel=ylabel, xlabel=xlabel)
    if type(filepath) is not type(None):
        plt.savefig(filepath+"_flash_distribution.svg")

def compare_cell_dist_df(dfs,bins,colors,labels,alpha, ylabel="",xlabel="",metric='change_modulation',filepath=None):
    dists = []
    for index, df in enumerate(dfs):     
        dist = df.groupby(['ophys_experiment_id','cell']).mean()[metric].values
        dists.append(dist)
    compare_dist(dists,bins,colors,labels,alpha, ylabel=ylabel, xlabel=xlabel)
    if type(filepath) is not type(None):
        plt.savefig(filepath+"_cell_distribution.svg")

def compare_session_dist_df(dfs,bins,colors,labels,alpha, ylabel="",xlabel="",metric='change_modulation',filepath=None):
    dists = []
    for index, df in enumerate(dfs):     
        dist = df.groupby(['ophys_experiment_id']).mean()[metric].values
        dists.append(dist)
    compare_dist(dists,bins,colors,labels,alpha, ylabel=ylabel, xlabel=xlabel)
    if type(filepath) is not type(None):
        plt.savefig(filepath+"_session_distribution.svg")

def compare_dist(dists,bins,colors,labels,alpha,ylabel="",xlabel=""):
    plt.figure(figsize=(3,3))
    for index,dist in enumerate(dists):
        counts,edges = np.histogram(dist,bins[index])
        centers = edges[0:-1] + np.diff(edges)/2
        plt.bar(centers, counts/np.sum(counts)/np.diff(edges)[0],width=np.diff(edges),color=colors[index],alpha=alpha[index], label=labels[index])
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(-1,1)
    plt.gca().axvline(0,linestyle='--',color='k',alpha=0.3)
    plt.tight_layout()

def compare_means(groups,colors,labels,ylim,axislabels):
    plt.figure(figsize=(3,3))
    for index, g in enumerate(groups):
        for eldex,el in enumerate(g):
            plt.plot(index, np.mean(el),'o',color=colors[eldex])
            plt.plot([index,index],[np.mean(el)-np.std(el)/np.sqrt(len(el)), np.mean(el)+np.std(el)/np.sqrt(len(el))],'-',color=colors[eldex])
    plt.xticks(range(0,len(groups)),labels)
    plt.xlim(-1,len(groups))
    plt.ylim(ylim)
    plt.ylabel(axislabels)

def compare_groups(group1,group2,labels):
    compare_dist([np.hstack(np.hstack(group1[0])),np.hstack(np.hstack(group2[0]))],[50,50],['k','r'],labels,[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
    compare_dist([np.hstack(group1[1]),np.hstack(group2[1])],[30,20],['k','r'],labels,[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
    compare_dist([np.hstack(group1[2]),np.hstack(group2[2])],[5,5],['k','r'],labels,[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
    compare_means([[np.hstack(np.hstack(group1[0])),np.hstack(np.hstack(group2[0]))],[np.hstack(group1[1]),np.hstack(group2[1])],[np.hstack(group1[2]),np.hstack(group2[2])]],['k','r'],['Flash','Cell','Session'],[0,.15],'Change Modulation')

def compare_groups_df(dfs,labels,metric='change_modulation', xlabel="Change Modulation",alpha=0.5, nbins=[5,50,50],savename=None,nboots=1000,plot_nice=False):
    if type(savename) is not type(None):
        filepath = '/home/alex.piet/codebase/behavior/doc/figures/change_modulation_figures/'+savename
        if metric is not 'change_modulation':
            filepath = filepath + "_"+metric
    else:
        filepath = None

    numdfs = len(dfs)
    colors = sns.color_palette(n_colors=len(dfs))

    if not plot_nice:
        for index, df in enumerate(dfs):
            plot_manifest_change_modulation_df(df,plot_cells=False,titlestr=labels[index],filepath=filepath,metric=metric)
    if not plot_nice:
        compare_session_dist_df(dfs, np.tile(nbins[0],numdfs),colors,labels,np.tile(alpha,numdfs),xlabel="Session Avg."+xlabel,ylabel="Prob",filepath=filepath,metric=metric)
    compare_cell_dist_df(dfs,    np.tile(nbins[1],numdfs),colors,labels,np.tile(alpha,numdfs),xlabel="Cell Avg. "+xlabel,ylabel="Prob",filepath=filepath,metric=metric) 
    compare_flash_dist_df(dfs,   np.tile(nbins[2],numdfs),colors,labels,np.tile(alpha,numdfs),xlabel="Flash "+xlabel,ylabel="Prob",filepath=filepath,metric=metric) 
    if plot_nice:
        compare_means_df(dfs, labels,filepath=filepath,metric=metric,nboots=nboots,plot_nice=plot_nice,labels=['Flash','Cell'])
    else:
        compare_means_df(dfs, labels,filepath=filepath,metric=metric,nboots=nboots,plot_nice=plot_nice)

def annotate_stage(df):
    df['image_set'] = [x[15] for x in df['stage'].values]
    df['active'] = [ (x[6] in ['1','3','4','6']) for x in df['stage'].values]
    df['stage_num'] = [x[6] for x in df['stage'].values]
    return df

def compare_means_df(dfs,df_labels,metric='change_modulation',ylabel='Change Modulation',labels=['Flash','Cell','Session'],ylim=[0,.2],filepath=None,titlestr="",nboots=1000,plot_nice=False):
    plt.figure(figsize=(3,3))
    colors = sns.color_palette(n_colors=len(dfs))

    df_flash_boots = []
    df_cell_boots = []
    if plot_nice:
        offset = 0
    else:
        offset = 0.1
    for index, df in enumerate(dfs):
        dists = [df[metric].values,  df.groupby(['ophys_experiment_id','cell']).mean()[metric].values,df.groupby(['ophys_experiment_id']).mean()[metric].values]
        if nboots > 0:
            flash_boots = bootstrap_df(df,nboots,metric=metric)
            df_flash_boots.append(flash_boots[2])
            plt.plot(offset+index*0.05,flash_boots[0],'o',color=colors[index],alpha=0.5)
            plt.plot([offset+index*0.05,offset+index*0.05],[flash_boots[0]-flash_boots[1], flash_boots[0]+flash_boots[1]],'-',color=colors[index],alpha=0.5)

            cell_boots = bootstrap_df(df.groupby(['ophys_experiment_id','cell']).mean().reset_index(),nboots,levels=['root','ophys_experiment_id'],metric=metric)
            df_cell_boots.append(cell_boots[2])
            plt.plot(1+offset+index*0.05,cell_boots[0],'o',color=colors[index],alpha=0.5)
            plt.plot([1+offset+index*0.05,1+offset+index*0.05],[cell_boots[0]-cell_boots[1], cell_boots[0]+cell_boots[1]],'-',color=colors[index],alpha=0.5)
        
        if not plot_nice:
            for ddex,dist in enumerate(dists):
                if ddex == 0:
                    plt.plot(ddex, np.mean(dist),'o',color=colors[index],label=df_labels[index])
                else:
                    plt.plot(ddex, np.mean(dist),'o',color=colors[index])
                plt.plot([ddex,ddex],[np.mean(dist)-np.std(dist)/np.sqrt(len(dist)), np.mean(dist)+np.std(dist)/np.sqrt(len(dist))],'-',color=colors[index])
    plt.xticks(range(0,len(labels)),labels)
    plt.xlim(-1,len(labels))
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(titlestr)
    if np.sum(df_flash_boots[0] > df_flash_boots[1])/len(df_flash_boots[0]) > (1-0.05):
        plt.plot(offset,plt.gca().get_ylim()[1]*.9,'k*',markersize=10)
    if np.sum(df_cell_boots[0] > df_cell_boots[1])/len(df_cell_boots[0]) > (1-0.05):
        plt.plot(1+offset,plt.gca().get_ylim()[1]*.9,'k*',markersize=10)

    print("Avg. Flash Difference: " + str(np.mean(df_flash_boots[0]-df_flash_boots[1])))     
    print("Avg. Cell Difference:  " + str(np.mean(df_cell_boots[0]-df_cell_boots[1])))    
    plt.title('Avg Flash Effect '+  str(round(np.mean(df_flash_boots[0]-df_flash_boots[1]),4)))
    plt.tight_layout()
    if type(filepath) is not type(None):
        plt.savefig(filepath+"_mean_comparison.svg")
    
def bootstrap_df(df, nboot,metric='change_modulation',levels=['root', 'ophys_experiment_id','cell']):
    h = hb.HData(df, levels)
    nsamples = len(df)
    bootstats = np.empty(nboot)
    metric_vals = h.df[metric].values

    for ind_rep in tqdm(range(nboot), position=0):
        inds_list = []
        hb.resample_recursive(h.root, inds_list)
        bootstats[ind_rep] = np.mean(metric_vals[inds_list])
    
    return np.mean(bootstats), np.std(bootstats), bootstats

def get_variance_by_level(df, levels=['ophys_experiment_id','cell'],metric='change_modulation'):
    cell_var = np.mean(df.groupby(levels)[metric].var())
    session_var = np.mean(df.groupby(levels)[metric].mean().groupby(levels[0:1]).var())
    pop_var = np.var(df.groupby(levels)[metric].mean().groupby(levels[0:1]).mean())
    pop_mean = np.mean(df.groupby(levels)[metric].mean().groupby(levels[0:1]).mean())
    var_vec = [cell_var, session_var, pop_var]
    return var_vec, np.sum(var_vec)/pop_mean



def block_to_mean_dff(df):
    return df['mean_response'][0:8].values

def get_cell_psth(cell,session):
    fr = session.flash_response_df.query('pref_stim')
    image_name = fr.iloc[0].image_name
    cell_fr = fr.query('cell_specimen_id == @cell').groupby('block_index').apply(block_to_mean_dff)
    cell_fr = [x for x in list(cell_fr) if len(x) == 8]  
    if len(cell_fr) > 0:
        cell_fr =  np.mean(np.vstack(cell_fr),0)
    else:
        cell_fr = []
    num_blocks =  len(fr.query('cell_specimen_id ==@cell')['block_index'].unique())
    trial_fr = session.trial_response_df.query('pref_stim & cell_specimen_id == @cell & go ')['dff_trace'].mean() 
    trial_fr_timestamps = session.trial_response_df.iloc[0].dff_trace_timestamps - session.trial_response_df.iloc[0].change_time
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],'mean_response_trace':[],'dff_trace':[],'dff_trace_timestamps':[],'preferred_stim':[],'number_blocks':[]})
    d = {
        'ophys_experiment_id':session.metadata['ophys_experiment_id'],
        'stage':session.metadata['stage'],
        'cell':cell,
        'imaging_depth':session.metadata['imaging_depth'],
        'mean_response_trace':cell_fr,
        'dff_trace':trial_fr,
        'dff_trace_timestamps':trial_fr_timestamps,
        'preferred_stim':image_name,
        'number_blocks':num_blocks
        }
    df = df.append(d,ignore_index=True)
    return df

def get_session_psth(session):
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],'mean_response_trace':[],'dff_trace':[],'dff_trace_timestamps':[],'preferred_stim':[],'number_blocks':[]})
    cellids = session.flash_response_df['cell_specimen_id'].unique()
    for index, cell in enumerate(cellids):
        cell_df = get_cell_psth(cell,session)
        df = df.append(cell_df,ignore_index=True)
    return df

def get_average_psth(session_ids):
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],'mean_response_trace':[],'dff_trace':[],'dff_trace_timestamps':[],'preferred_stim':[],'number_blocks':[]})
    for index, session_id in tqdm(enumerate(session_ids)):
        session = pgt.get_data(session_id)
        session_df = get_session_psth(session)
        df = df.append(session_df,ignore_index=True)
    df = annotate_stage(df)
    return df 

def get_all_df(path='/home/alex.piet/codebase/allen/all_slc_df.csv', force_recompute=False):
    try:
        all_df =pd.read_csv(filepath_or_buffer = path)
    except:
        if force_recompute:
            all_df, *all_list = manifest_change_modulation(pgt.get_slc_session_ids())
            all_df = annotate_stage(all_df)
            all_df.to_csv(path_or_buf=path)
        else:
            raise Exception('file not found: '+path)
    return all_df

def get_all_exp_df(path='/home/alex.piet/codebase/allen/all_slc_exp_df.pkl',force_recompute=False):
    try:
        all_exp_df =pd.read_pickle(path)
    except:
        if force_recompute:
            all_exp_df = get_average_psth(pgt.get_slc_session_ids())
            all_exp_df.to_pickle(path)
        else:
            raise Exception('file not found: ' + path)
    return all_exp_df

def compare_exp_groups(all_df,queries, labels):
    dfs=[]
    for q in queries:
        dfs.append(all_df.query(q))
    plot_mean_trace(dfs,labels)

def plot_mean_trace(dfs, labels):    
    plt.figure()
    colors = sns.color_palette(n_colors=2)
    for index, df in enumerate(dfs):
        plt.plot(df.iloc[0]['dff_trace_timestamps'], df['dff_trace'].mean()-np.min(df['dff_trace'].mean()), color=colors[index], alpha=0.5,label=labels[index])
    plt.ylim(0,.2)
    plt.ylabel('Average PSTH (df/f)')
    plt.xlabel('# Repetition in Block')
    plt.legend()


def plot_cell_mean_trace(exp_df, cell,titlestr='',ophys_experiment_id = None):
    plt.figure(figsize=(6,3))
    for i in np.arange(-6,11):
        plt.axvspan(i*.75,i*.75+.25,color='k', alpha=0.1)

    plt.axvspan(0,.25,color='r', alpha=0.1)
    plt.axvspan(9*.75,9*.75+.25,color='r', alpha=0.1)
    if type(ophys_experiment_id) == type(None):
        colors = sns.color_palette(n_colors=6)
        for i in range(0,len(exp_df.query('cell == @cell'))):
            plt.plot(exp_df.iloc[0].dff_trace_timestamps, exp_df.query('cell == @cell').iloc[i]['dff_trace'],'-',color=colors[i],label=exp_df.query('cell == @cell').iloc[i]['stage'])
        plt.legend()
    else:
        plt.plot(exp_df.iloc[0].dff_trace_timestamps, exp_df.query('(ophys_experiment_id == @ophys_experiment_id) & (cell == @cell)').iloc[0]['dff_trace'],'k-')
    plt.ylabel('Average PSTH (df/f)')
    plt.xlabel('Time since image change (s)')
    plt.title(titlestr)

    plt.tight_layout()

def plot_top_n_cells(all_df,all_exp_df,query, top_n=1, metric='change_modulation',show_all_sessions=False):
    if top_n > 0:
        for i in range(0, top_n):
            plot_top_cell(all_df,all_exp_df,query, top=i, metric=metric,show_all_sessions=show_all_sessions)
    else:
        for i in range(top_n,0):
            plot_top_cell(all_df,all_exp_df,query, top=i, metric=metric,show_all_sessions=show_all_sessions)

def plot_top_cell(all_df, all_exp_df,query, top=0, metric='change_modulation',show_all_sessions=False):
    cell_session, cell_id = get_top_cell(all_df, query,metric=metric, top=top)

    if show_all_sessions:
        titlestr = 'Session '+str(cell_session) +'  Cell '+str(cell_id)
        plot_cell_mean_trace(all_exp_df,cell_id,titlestr=titlestr)
    else:
        mean_metric = all_df.query('(cell==@cell_id) & (ophys_experiment_id ==@cell_session)').mean()[metric]
        stage = all_exp_df.query('ophys_experiment_id == @cell_session').iloc[0]['stage']
        num_blocks = all_exp_df.query('ophys_experiment_id == @cell_session').iloc[0]['number_blocks']
        titlestr = 'Session '+str(cell_session) +'  Cell '+str(cell_id)+'\n '+stage + '  CM:'+str(round(mean_metric,2))+'  # Changes:'+str(num_blocks.astype(int))
        plot_cell_mean_trace(all_exp_df,cell_id,titlestr=titlestr,ophys_experiment_id =cell_session)
    
def get_top_cell(df, query, metric='change_modulation', top=0):
    if len(query) == 0:
        cell = df.groupby(['cell','ophys_experiment_id']).mean().sort_values(by=metric).iloc[top]
    else:
        cell = df.query(query).groupby(['cell']).mean().sort_values(by=metric).iloc[top]
    return cell.name[1].astype(int), cell.name[0].astype(int)




#############################
def get_slc_dfs(file1='slc_df.pkl', file2='slc_full_df.pkl',dir_path='/home/alex.piet/codebase/allen/',force_recompute=False):
    try:
        slc_df =pd.read_pickle(dir_path+file1)
        slc_cell_df =pd.read_pickle(dir_path+file2)
    except:
        if force_recompute:
            slc_df, slc_cell_df = build_slc_dfs(pgt.get_slc_session_ids())
            slc_df.to_pickle(path=dir_path+file1)
            slc_cell_df.to_pickle(path=dir_path+file2)
        else:
            raise Exception('files not found: '+dir_path+' '+file1+' '+file2)
    return slc_df,slc_cell_df

def build_slc_dfs(ids):
    slc_df, *list_stuff = manifest_change_modulation(ids)
    slc_df = annotate_stage(slc_df)
    slc_cell_df =slc_df.copy()
    return slc_df, slc_cell_df

def get_top_cells(slc_df,n,query='good_response & reliable_cell & good_block',negative_cells=True,metric='change_modulation'):
    return slc_df.query(query).groupby('cell')[metric].mean().sort_values(ascending=negative_cells).head(n).index.values
    
def plot_mean_trace(slc_df,query,plot_each =True):
    plt.figure(figsize=(6,3))
    for i in np.arange(0,11):
        plt.axvspan(i*.75,i*.75+.25,color='k', alpha=0.1)

    if plot_each:
        for index, row in slc_df.query(query).iterrows():
            plt.plot(row.timestamps[0:232]-row.timestamps[0], row.dff_trace[0:232],alpha=0.3)
    plt.plot(slc_df.query(query).iloc[0]['timestamps']-slc_df.query(query).iloc[0]['timestamps'][0],slc_df.query(query)['dff_trace'].mean(), 'k-',linewidth=2)
    if len(slc_df.query(query)['cell'].unique()) ==1:
        session = str(len(slc_df.query(query)['ophys_experiment_id'].unique()))
        cell = str(slc_df.query(query)['cell'].iloc[0].astype(int))
        blocks = str(len(slc_df.query(query)))
        titlestr = 'Cell: '+cell+"\n # Changes: "+blocks+", from "+session+" sessions"
    else:
        titlestr = str(len(slc_df.query(query)['cell'].unique())) +" Cells, from "+ str(len(slc_df.query(query)['ophys_experiment_id'].unique()))+" Sessions"

    plt.ylabel('Average PSTH (df/f)')
    plt.xlabel('Time since image change (s)')
    plt.title(titlestr)
    plt.tight_layout()

def plot_trace_image(slc_df, query):
    plt.figure()
    plt.imshow(np.vstack(slc_df.query(query)['dff_trace']),cmap='plasma',aspect='auto')



