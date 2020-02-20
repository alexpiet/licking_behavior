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
import os

def get_cell_df(cell_id, session):
    return session.flash_response_df[session.flash_response_df['cell_specimen_id'] == cell_id]

def test_cell_reliability(cell_flash_df, pval=0.05, percent=0.25):
    return (np.sum(cell_flash_df['p_value'] < pval)/len(cell_flash_df)) > percent

def cell_change_modulation(cell, session):
    '''
        Computes CM for a single cell
        good_response, metric within -1,1 bounds
        good_response_dc, metric within -1,1 bounds
        good_block, at least 10 flash repetitions
        reliable_cell, at least 25% of repetitions of pref_stim are significant 
        real_response, sum of both responses is greater than 0.1


    '''
    cell_flash_df = get_cell_df(cell,session)
    cell_flash_df = cell_flash_df[cell_flash_df['pref_stim']]

    cms = []
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],
        'change_modulation':[],'block_number':[],'change_modulation_base':[],'pref_stim':[], 
        'reliable_cell':[], 'good_response':[], 'dff_trace':[],'good_block':[],'good_response_base':[],
        'timestamps':[],'mean_response_0':[],'mean_response_9':[],'mean_response_0_base':[],'mean_response_9_base':[],
        'change_time':[],'start_stimulus_presentations_id':[]})

    reliable_cell = test_cell_reliability(cell_flash_df,percent=0.25)  
    blocks = cell_flash_df['block_index'].unique()
    timestamps = session.ophys_timestamps
    cell_dff_trace = session.dff_traces.query('cell_specimen_id ==@cell').iloc[0]['dff']
    for block in blocks:        
        block_df = cell_flash_df[cell_flash_df['block_index'] == block] 
        if len(block_df) >= 10:
            this_change= block_df.iloc[0]['mean_response']
            this_change_base = block_df.iloc[0]['baseline_response']
            this_non= block_df.iloc[9]['mean_response']
            this_non_base = block_df.iloc[9]['baseline_response']
            this_cm = (this_change - this_non)/(this_change+this_non)
            this_cm_base = ((this_change-this_change_base) - (this_non-this_non_base))/((this_change-this_change_base)+(this_non - this_non_base))
            dff_trace = cell_dff_trace[(timestamps > block_df.iloc[0]['start_time']-3.0) & (timestamps < block_df.iloc[9]['start_time']+0.75)]
            ophys_timestamps = timestamps[(timestamps > block_df.iloc[0]['start_time']-3.0) & (timestamps < block_df.iloc[9]['start_time']+0.75)]
        else:
            this_change= block_df.iloc[0]['mean_response']
            this_change_base = block_df.iloc[0]['baseline_response']
            this_non= block_df.iloc[-1]['mean_response']
            this_non_base = block_df.iloc[-1]['baseline_response']
            this_cm = (this_change - this_non)/(this_change+this_non)
            this_cm_base = ((this_change-this_change_base) - (this_non-this_non_base))/((this_change-this_change_base)+(this_non - this_non_base))
            dff_trace = cell_dff_trace[(timestamps > block_df.iloc[0]['start_time']-3.0) & (timestamps < block_df.iloc[-1]['start_time']+0.75)]
            ophys_timestamps = timestamps[(timestamps > block_df.iloc[0]['start_time']-3.0) & (timestamps < block_df.iloc[-1]['start_time']+0.75)]

        good_response = (this_cm > -1) & (this_cm < 1) &(this_cm_base > -1) & (this_cm_base < 1)
        good_response_base = (this_cm_base > -1) & (this_cm_base < 1)
        good_block = len(block_df) >= 10
        if len(dff_trace) < 325:
            good_block = False       
        if good_response & good_block:
            cms.append(this_cm)

        d = {'ophys_experiment_id':session.metadata['ophys_experiment_id'],'stage':session.metadata['session_type'],
            'cell':cell,'imaging_depth':session.metadata['imaging_depth'],'change_modulation':this_cm,
            'block_number':block,'change_modulation_base':this_cm_base,'pref_stim':cell_flash_df.iloc[0]['image_name'],
            'reliable_cell':reliable_cell,'good_response':good_response,'dff_trace':dff_trace[0:325],'good_block':good_block,
            'good_response_base':good_response_base,'timestamps':ophys_timestamps[0:325],'mean_response_0':this_change, 'mean_response_9':this_non,
            'mean_response_0_base':this_change_base, 'mean_response_9_base':this_non_base,'change_time':block_df.iloc[0]['start_time'], 
            'start_stimulus_presentations_id':block_df.iloc[0]['stimulus_presentations_id']} # was 232 on timestamp index
        df = df.append(d,ignore_index=True)

    return df, cms

def session_change_modulation(id):
    '''
        Computes CM for a single session
    ''' 
    session = pgt.get_data(id)
    pgt.get_stimulus_response_df(session)

    cells = session.flash_response_df['cell_specimen_id'].unique()
    all_cms = []
    mean_cms = []
    var_cms = []
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],
        'change_modulation':[],'block_number':[],'change_modulation_base':[],'pref_stim':[], 
        'reliable_cell':[], 'good_response':[], 'dff_trace':[],'good_block':[],'good_response_base':[],
        'timestamps':[],'mean_response_0':[],'mean_response_9':[],'mean_response_0_base':[],'mean_response_9_base':[],
        'change_time':[],'start_stimulus_presentations_id':[]})
    for cell in cells:
        cell_df,cm = cell_change_modulation(cell,session)
        if len(cm) > 0:
            all_cms.append(cm)
            mean_cms.append(np.mean(cm))
            var_cms.append(np.var(cm))
            df = df.append(cell_df,ignore_index=True)
    session_mean = np.mean(mean_cms)
    session_var = np.var(mean_cms)
    session.cache_clear() 
    return df, all_cms,mean_cms,var_cms, session_mean, session_var

def manifest_change_modulation(ids,dc_offset=0.05,load_file=True):
    '''
        Takes a list of behavior_session_ids
        real_response, sum of both responses is greater than 0.1
    '''
    if load_file:
        print('Will load session files if they exist')
    all_cms = []
    mean_cms = []
    var_cms = []
    session_means = []
    session_vars = []
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],
        'change_modulation':[],'block_number':[],'change_modulation_base':[],'pref_stim':[], 
        'reliable_cell':[], 'good_response':[], 'dff_trace':[],'good_block':[],'good_response_base':[],
        'timestamps':[],'mean_response_0':[],'mean_response_9':[],'mean_response_0_base':[],'mean_response_9_base':[],
        'change_time':[],'start_stimulus_presentations_id':[]})
    for id in tqdm(ids):
        try:
            had_file = False
            filepath='/home/alex.piet/codebase/allen/data/session_files/change_modulation_df_'+str(id)+'.h5'
            if load_file & os.path.exists(filepath):
                session_df =pd.read_hdf(filepath,'df')
                cell_cms = []
                cell_mean_cms = []
                cell_var_cms = []
                session_mean = []
                session_var = []
                had_file=True
            else:
                session_df, cell_cms,cell_mean_cms,cell_var_cms, session_mean,session_var = session_change_modulation(id)
        except:
            print('crash '+str(id))
        else:
            all_cms.append(cell_cms)
            mean_cms.append(cell_mean_cms)
            var_cms.append(cell_var_cms)
            session_means.append(session_mean)
            session_vars.append(session_var)
            df = df.append(session_df,ignore_index=True)
            # Save out session df
            if not had_file:
                filepath='/home/alex.piet/codebase/allen/data/session_files/change_modulation_df_'+str(id)+'.h5'
                session_df.to_hdf(filepath,key='df',mode='w')
    df['good_block'] = df['good_block'] == 1.0
    df['good_response'] = df['good_response'] == 1.0
    df['good_response_base'] = df['good_response_base'] == 1.0
    df['reliable_cell'] = df['reliable_cell'] == 1.0
    df['change_modulation_dc'] = (df['mean_response_0']+dc_offset-(df['mean_response_9']+dc_offset))/(df['mean_response_0']+dc_offset+df['mean_response_9']+dc_offset) 
    df['good_response_dc'] = (df['change_modulation_dc'] > -1 ) & (df['change_modulation_dc'] < 1)
    df['change_modulation_base_dc'] = ((df['mean_response_0']-df['mean_response_0_base'])+dc_offset-((df['mean_response_9']-df['mean_response_9_base'])+dc_offset))/((df['mean_response_0']-df['mean_response_0_base'])+dc_offset+(df['mean_response_9']-df['mean_response_0'])+dc_offset) 
    df['good_response_base_dc'] = (df['change_modulation_base_dc'] > -1 ) & (df['change_modulation_base_dc'] < 1)
    df['real_response'] = (df['mean_response_0'] + df['mean_response_9']) > 0.1
    df = annotate_stage(df)


    return df, all_cms, mean_cms, var_cms,session_means, session_vars, np.mean(session_means), np.var(session_means)

def plot_manifest_change_modulation_df(df,box_plot=True,plot_cells=True,metric='change_modulation',titlestr="",filepath=None):
    # Set up some general things
    plt.figure()
    count = 0
    colors = sns.color_palette(n_colors=len(df['ophys_experiment_id'].unique()))
    
    # Order Sessions
    this_session_means = df.groupby(['ophys_experiment_id','cell']).mean()[metric].groupby('ophys_experiment_id').mean().values
    session_ids = df.groupby(['ophys_experiment_id','cell']).mean()[metric].groupby('ophys_experiment_id').mean().index.values
    session_means_sorted = sorted(this_session_means)
    session_ids_sorted = [x for _,x in sorted(zip(this_session_means,session_ids))]

    # Iterate over sessions
    for session_index, session_id in enumerate(session_ids_sorted):
        if plot_cells:
            # Sort Cells in this session
            session_cells = df[df['ophys_experiment_id'] == session_id].groupby(['cell']).mean()[metric].sort_values()
            session_cell_means = session_cells.values
            session_cell_ids = session_cells.index.values

            # Plot each cell/flash pair in this session 
            for index,this_cell in enumerate(session_cell_ids):
                this_cell_cms = df.query('ophys_experiment_id == @session_id & cell ==@this_cell')[metric].values
                if box_plot:
                    bplot = plt.gca().boxplot(this_cell_cms,showfliers=False,positions=[count],
                        widths=1,patch_artist=True,showcaps=False)
                    for whisker in bplot['whiskers']:
                        whisker.set_color(colors[session_index])
                    bplot['medians'][0].set_color(colors[session_index])
                    for patch in bplot['boxes']:
                        patch.set_facecolor(colors[session_index])
                        patch.set_edgecolor(colors[session_index])
                else:
                    plt.plot(np.repeat(count,len(this_cell_cms)), this_cell_cms,'o',color=colors[session_index])
                plt.plot(count, session_cell_means[index],'ko',zorder=5) 
                count +=1

            # Plot session mean
            plt.plot([count-len(session_cell_ids),count-1],[session_means_sorted[session_index], 
                    session_means_sorted[session_index]],color=colors[session_index],label=str(session_index),linewidth=1,zorder=10)
            plt.plot([count-len(session_cell_ids),count-1],[session_means_sorted[session_index], 
                    session_means_sorted[session_index]],'k',linewidth=4,zorder=10)
        else:
            session_cells = df[df['ophys_experiment_id'] == session_id].groupby(['cell']).mean()[metric].sort_values()
            session_cell_means = session_cells.values
            bplot= plt.gca().boxplot(session_cell_means,showfliers=False,positions=[session_index],
                widths=1,patch_artist=True,showcaps=False)
            for whisker in bplot['whiskers']:
                whisker.set_color(colors[session_index])
            bplot['medians'][0].set_color(colors[session_index])
            for patch in bplot['boxes']:
                patch.set_facecolor(colors[session_index])
                patch.set_edgecolor(colors[session_index])
            plt.plot([session_index-0.4,session_index+0.4],[session_means_sorted[session_index], 
                session_means_sorted[session_index]],color=colors[session_index],label=str(session_index),linewidth=1,zorder=10)
            plt.plot([session_index-0.4,session_index+0.4],[session_means_sorted[session_index], 
                session_means_sorted[session_index]],'k',linewidth=4,zorder=10)

    # clean up plot
    plt.ylim([-1,1])
    plt.gca().axhline(0,linestyle='--',color='k',alpha=1)
    plt.ylabel(metric)
    plt.title(titlestr)
    if plot_cells:
        plt.xlabel('Cell #')
        plt.xticks(np.arange(0,count,50),np.arange(0,count,50).astype('str'))
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
                    bplot = plt.gca().boxplot(this_cell_cms,showfliers=False,positions=[count],
                        widths=1,patch_artist=True,showcaps=False)
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
            plt.plot([count-len(session_cell_cms),count-1],[session_means_sorted[session_index], 
                session_means_sorted[session_index]],color=colors[session_index],label=str(session_index),linewidth=1,zorder=10)
            plt.plot([count-len(session_cell_cms),count-1],[session_means_sorted[session_index], 
                session_means_sorted[session_index]],'k',linewidth=4,zorder=10)
        else:
            bplot= plt.gca().boxplot(cell_mean_cms_sorted[session_index],showfliers=False,
                positions=[session_index],widths=1,patch_artist=True,showcaps=False)
            for whisker in bplot['whiskers']:
                whisker.set_color(colors[session_index])
            bplot['medians'][0].set_color(colors[session_index])
            for patch in bplot['boxes']:
                patch.set_facecolor(colors[session_index])
                patch.set_edgecolor(colors[session_index])
            plt.plot([session_index-0.4,session_index+0.4],[session_means_sorted[session_index], 
                session_means_sorted[session_index]],color=colors[session_index],label=str(session_index),linewidth=1,zorder=10)
            plt.plot([session_index-0.4,session_index+0.4],[session_means_sorted[session_index], 
                session_means_sorted[session_index]],'k',linewidth=4,zorder=10)

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
    shuffle_df = shuffle_df.apply(np.random.permutation,axis=axis)
    return shuffle_df
    
def bootstrap_session_cell_modulation_df(df,numboots):
    shuffle_df = shuffle(df)
    for i in range(0,numboots):
        this_df = shuffle(df)
        this_df['cell'] = this_df['cell'] + 10000*(i+1)
        shuffle_df = shuffle_df.append(this_df,ignore_index=True)
    return shuffle_df

def compare_flash_dist_df(dfs,colors,labels,alpha, ylabel="",xlabel="",metric='change_modulation',filepath=None,bins=None):
    dists = []
    for index, df in enumerate(dfs):     
        dist = df[metric].values
        dists.append(dist)
    compare_dist(dists,colors,labels,alpha, ylabel=ylabel, xlabel=xlabel,bins=bins)
    if type(filepath) is not type(None):
        plt.savefig(filepath+"_flash_distribution.svg")

def compare_cell_dist_df(dfs,colors,labels,alpha, ylabel="",xlabel="",metric='change_modulation',filepath=None,bins=None):
    dists = []
    for index, df in enumerate(dfs):     
        dist = df.groupby(['ophys_experiment_id','cell']).mean()[metric].values
        dists.append(dist)
    compare_dist(dists,colors,labels,alpha, ylabel=ylabel, xlabel=xlabel,bins=bins)
    if type(filepath) is not type(None):
        plt.savefig(filepath+"_cell_distribution.svg")

def compare_session_dist_df(dfs,colors,labels,alpha, ylabel="",xlabel="",metric='change_modulation',filepath=None,bins=None):
    dists = []
    for index, df in enumerate(dfs):     
        dist = df.groupby(['ophys_experiment_id']).mean()[metric].values
        dists.append(dist)
    compare_dist(dists,colors,labels,alpha, ylabel=ylabel, xlabel=xlabel,bins=bins)
    if type(filepath) is not type(None):
        plt.savefig(filepath+"_session_distribution.svg")

def compare_dist(dists,colors,labels,alpha,ylabel="",xlabel="",bins=None):
    plt.figure(figsize=(3,3))
    # Make bins dynamic here
    if bins is None:
        bins = [int(np.floor(len(x)/50)) if len(x) < 500 else int(np.floor(len(x)/100)) if len(x) < 5000 else int(np.floor(len(x)/1000)) for x in dists]
    for index,dist in enumerate(dists):
        counts,edges = np.histogram(dist,bins[index])
        centers = edges[0:-1] + np.diff(edges)/2
        plt.bar(centers, counts/np.sum(counts)/np.diff(edges)[0],color=colors[index],alpha=alpha[index], label=labels[index],width = np.diff(edges))
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
    compare_dist([np.hstack(np.hstack(group1[0])),np.hstack(np.hstack(group2[0]))],['k','r'],labels,[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
    compare_dist([np.hstack(group1[1]),np.hstack(group2[1])],['k','r'],labels,[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
    compare_dist([np.hstack(group1[2]),np.hstack(group2[2])],['k','r'],labels,[0.5,0.5],ylabel='Prob/Bin',xlabel='Change Modulation')
    compare_means([[np.hstack(np.hstack(group1[0])),np.hstack(np.hstack(group2[0]))],[np.hstack(group1[1]),np.hstack(group2[1])],[np.hstack(group1[2]),np.hstack(group2[2])]],['k','r'],['Flash','Cell','Session'],[0,.15],'Change Modulation')

def compare_groups_df(dfs,labels,metric='change_modulation', xlabel="Change Modulation",alpha=0.3, nbins=None,savename=None,nboots=1000,plot_nice=False,ylim = None,ci=False):
    if type(savename) is not type(None):
        filepath = '/home/alex.piet/codebase/behavior/change_modulation/'+savename
        if metric is not 'change_modulation':
            filepath = filepath + "_"+metric
    else:
        filepath = None

    numdfs = len(dfs)
    colors = sns.color_palette(n_colors=len(dfs))
    if nbins is None:
        nbins = [np.tile(10,numdfs),np.tile(50,numdfs),np.tile(50,numdfs)]

    if not plot_nice:
        for index, df in enumerate(dfs):
            plot_manifest_change_modulation_df(df,plot_cells=False,titlestr=labels[index],filepath=filepath,metric=metric)
    if not plot_nice:
        compare_session_dist_df(dfs,colors,labels,np.tile(alpha,numdfs),xlabel="Session Avg."+xlabel,ylabel="Prob",filepath=filepath,metric=metric,bins=nbins[0])
    compare_cell_dist_df(dfs, colors,labels,np.tile(alpha,numdfs),xlabel="Cell Avg. "+xlabel,ylabel="Prob",filepath=filepath,metric=metric,bins=nbins[1]) 
    compare_flash_dist_df(dfs,colors,labels,np.tile(alpha,numdfs),xlabel="Flash "+xlabel,ylabel="Prob",filepath=filepath,metric=metric,bins=nbins[2]) 
    if plot_nice:
        compare_means_df(dfs, labels,filepath=filepath,metric=metric,nboots=nboots,plot_nice=plot_nice,labels=['Flash','Cell'],ylim = ylim,ci=ci)
    else:
        compare_means_df(dfs, labels,filepath=filepath,metric=metric,nboots=nboots,plot_nice=plot_nice,ylim = ylim,ci=ci)

def annotate_stage(df):
    df['image_set'] = [x[15] for x in df['stage'].values]
    df['active'] = [ (x[6] in ['1','3','4','6']) for x in df['stage'].values]
    df['stage_num'] = [x[6] for x in df['stage'].values]
    return df

def compare_means_df(dfs,df_labels,metric='change_modulation',ylabel='Change Modulation',labels=['Flash','Cell','Session'],ylim=None,filepath=None,titlestr="",nboots=1000,plot_nice=False,ci=True):
    plt.figure(figsize=(3,3))
    colors = sns.color_palette(n_colors=len(dfs))

    df_flash_boots = []
    df_cell_boots = []
    if plot_nice:
        offset = 0
    else:
        offset = 0
    boot_offset = 0.05
    for index, df in enumerate(dfs):
        dists = [df[metric].values,  df.groupby(['ophys_experiment_id','cell']).mean()[metric].values,df.groupby(['ophys_experiment_id']).mean()[metric].values]
        if nboots > 0:
            flash_boots = bootstrap_df(df,nboots,metric=metric)
            df_flash_boots.append(flash_boots[2])
            plt.plot(offset+index*boot_offset,flash_boots[0],'o',color=colors[index],alpha=0.5)
            if ci:
                plt.plot([offset+index*boot_offset,offset+index*boot_offset],[flash_boots[0]-2*flash_boots[1], flash_boots[0]+2*flash_boots[1]],'-',color=colors[index],alpha=0.5,label=df_labels[index])
            else:
                plt.plot([offset+index*boot_offset,offset+index*boot_offset],[flash_boots[0]-flash_boots[1], flash_boots[0]+flash_boots[1]],'-',color=colors[index],alpha=0.5,label=df_labels[index])

            cell_boots = bootstrap_df(df.groupby(['ophys_experiment_id','cell']).mean().reset_index(),nboots,levels=['root','ophys_experiment_id'],metric=metric)
            df_cell_boots.append(cell_boots[2])
            plt.plot(1+offset+index*boot_offset,cell_boots[0],'o',color=colors[index],alpha=0.5)
            if ci:
                plt.plot([1+offset+index*boot_offset,1+offset+index*boot_offset],[cell_boots[0]-2*cell_boots[1], cell_boots[0]+2*cell_boots[1]],'-',color=colors[index],alpha=0.5)
            else:
                plt.plot([1+offset+index*boot_offset,1+offset+index*boot_offset],[cell_boots[0]-cell_boots[1], cell_boots[0]+cell_boots[1]],'-',color=colors[index],alpha=0.5)
        
        if not plot_nice:
            for ddex,dist in enumerate(dists):
                if ddex == 0:
                    plt.plot(ddex, np.mean(dist),'o',color=colors[index])
                else:
                    plt.plot(ddex, np.mean(dist),'o',color=colors[index])
                plt.plot([ddex,ddex],[np.mean(dist)-np.std(dist)/np.sqrt(len(dist)), np.mean(dist)+np.std(dist)/np.sqrt(len(dist))],'-',color=colors[index])
    plt.xticks(range(0,len(labels)),labels)
    plt.xlim(-1,len(labels))
    if ylim is not None:
        plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.axhline(0,ls='--',color='k',alpha=0.2)
    plt.legend()
    plt.title(titlestr)
    yval = plt.gca().get_ylim()[1]*.9
    if len(dfs) > 1:
        if (np.sum(df_flash_boots[0] > df_flash_boots[1])/len(df_flash_boots[0]) > (1-0.05)) or (np.sum(df_flash_boots[0] > df_flash_boots[1])/len(df_flash_boots[0]) < 0.05):
            plt.plot(offset,yval,'k*',markersize=10)
        if (np.sum(df_cell_boots[0] > df_cell_boots[1])/len(df_cell_boots[0]) > (1-0.05)) or (np.sum(df_cell_boots[0] > df_cell_boots[1])/len(df_cell_boots[0]) < 0.05):
            plt.plot(1+offset,yval,'k*',markersize=10)

        print("Avg. Flash Difference: " + str(np.mean(df_flash_boots[0]-df_flash_boots[1])))     
        print("Avg. Cell Difference:  " + str(np.mean(df_cell_boots[0]-df_cell_boots[1])))    
        if not plot_nice:
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
        'stage':session.metadata['session_type'],
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

def get_all_df(filepath='/home/alex.piet/codebase/allen/data/change_modulation_df.h5', force_recompute=False,ids=[],savefile=False,load_files=True):
   
    if force_recompute:
        print('Recomputing Change Modulation df') 
        if len(ids) == 0:
            print('Rebuilding Manifest for Full Containers') 
            pgt.get_manifest(require_full_container=True,force_recompute=True)
            ids = pgt.get_session_ids()
        print('Starting Computation') 
        all_df, *all_list = manifest_change_modulation(ids,load_file = load_files)
        print('Finished Compiling Sessions') 

        if savefile:
            print('Attempting to Save') 
            try:
                all_df.to_hdf(filepath,key='df',mode='w')
            except:
                print('Could not save file')                
        return all_df
    try:
        all_df =pd.read_hdf(filepath,'df')
    except:
        raise Exception('file not found: '+filepath)
    return all_df

def get_all_exp_df(path='/home/alex.piet/codebase/allen/data/all_slc_exp_df.pkl',force_recompute=False):
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

def plot_mean_trace(dfs, labels):   # Is this redundant to plot_top_cell() 
    plt.figure()
    colors = sns.color_palette(n_colors=2)
    for index, df in enumerate(dfs):
        plt.plot(df.iloc[0]['dff_trace_timestamps'], df['dff_trace'].mean()-np.min(df['dff_trace'].mean()), color=colors[index], alpha=0.5,label=labels[index])
    plt.ylim(0,.2)
    plt.ylabel('Average PSTH (df/f)')
    plt.xlabel('# Repetition in Block')
    plt.legend()

def plot_cell_mean_trace(all_df, cell,titlestr='',ophys_experiment_id = None,show_all_blocks=False,metric='change_modulation'):
    plt.figure(figsize=(6,3))
    if type(ophys_experiment_id) == type(None):
        mean_metric = all_df.query('cell==@cell').mean()[metric]
        num_sessions = len(all_df.query('cell==@cell')['ophys_experiment_id'].unique())
        num_blocks = len(all_df.query('cell==@cell'))
        cre = all_df.query('cell==@cell').iloc[0].cre_line
        titlestr = '# Sessions: '+str(num_sessions) +'  Cell '+str(cell)+'  Cre: '+str(cre)+'\n   CM:'+str(round(mean_metric,2))+'  # Changes:'+str(num_blocks)
    else:
        mean_metric = all_df.query('(cell==@cell) & (ophys_experiment_id ==@ophys_experiment_id)').mean()[metric]
        stage = all_df.query('ophys_experiment_id == @ophys_experiment_id').iloc[0]['stage']
        num_blocks = len(all_df.query('cell==@cell & ophys_experiment_id == @ophys_experiment_id'))
        cre = all_df.query('cell==@cell').iloc[0].cre_line
        titlestr = 'Session '+str(ophys_experiment_id) +'  Cell '+str(cell)+'  Cre: '+str(cre)+'\n '+stage + '  CM:'+str(round(mean_metric,2))+'  # Changes:'+str(num_blocks)
    plt.title(titlestr)
    for i in np.arange(-6,11):
        plt.axvspan(i*.75,i*.75+.25,color='k', alpha=0.1)

    plt.axvspan(0,.25,color='r', alpha=0.1)
    plt.axhline(0,color='k',ls='-',alpha=0.1)

    # Plot each block
    if show_all_blocks:
        if type(ophys_experiment_id) == type(None):
            cell_df = all_df.query('cell==@cell')
        else:
            cell_df = all_df.query('cell==@cell & ophys_experiment_id ==@ophys_experiment_id')
        colors = sns.color_palette(n_colors=len(cell_df))
        for i in range(0,len(cell_df)):
            plt.plot(cell_df.iloc[i].timestamps-cell_df.iloc[i].timestamps[93], cell_df.iloc[i]['dff_trace'],'-',color=colors[i],alpha=0.5)

    # Plot Session Average
    if type(ophys_experiment_id) == type(None):
        sessions = all_df.query('cell == @cell')['ophys_experiment_id'].unique()
        stages = all_df.query('cell == @cell')['stage'].unique()
        colors = sns.color_palette(n_colors=len(stages))
        for index,val in enumerate(sessions):
            plt.plot(all_df.iloc[0].timestamps-all_df.iloc[0].timestamps[93], all_df.query('(ophys_experiment_id == @val) & (cell == @cell)')['dff_trace'].mean(),'-',color=colors[index],alpha=0.5,label=stages[index]+' Avg')
        plt.plot(all_df.iloc[0].timestamps-all_df.iloc[0].timestamps[93], all_df.query('cell == @cell')['dff_trace'].mean(),'k-',label='Avg.')
    else:
        # Plot Grand Average
        plt.plot(all_df.iloc[0].timestamps-all_df.iloc[0].timestamps[93], all_df.query('(ophys_experiment_id == @ophys_experiment_id) & (cell == @cell)')['dff_trace'].mean(),'k-',label='Avg.')
    plt.ylabel('Average PSTH (df/f)',fontsize=16)
    plt.xlabel('Time since image change (s)',fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(-3,7.5)
    plt.legend()
    plt.tight_layout()


def plot_top_cell(all_df, top=0, metric='change_modulation',show_all_sessions=False,show_all_blocks=False,median=False):
    if median:
        cell_session, cell_id = get_median_cell(all_df, metric=metric, top=top)
    else:
        cell_session, cell_id = get_top_cell(all_df, metric=metric, top=top)

    if show_all_sessions: # Plots average from each session
        plot_cell_mean_trace(all_df,cell_id,show_all_blocks=show_all_blocks,metric=metric)
    else:
        plot_cell_mean_trace(all_df,cell_id,ophys_experiment_id =cell_session,show_all_blocks=show_all_blocks,metric=metric)

def plot_top_n_cells(all_df, n=1, metric='change_modulation',show_all_sessions=False,show_all_blocks=False,median=False,start= 0):
    if n > 0:
        for i in range(start, start+n):
            plot_top_cell(all_df, top=i, metric=metric,show_all_sessions=show_all_sessions,show_all_blocks=show_all_blocks,median=median)
    else:
        for i in range(n+start,start):
            plot_top_cell(all_df, top=i, metric=metric,show_all_sessions=show_all_sessions,show_all_blocks=show_all_blocks,median=median)
    
def get_top_cell(df, metric='change_modulation', top=0):
    cell = df.groupby(['cell','ophys_experiment_id']).mean().sort_values(by=metric).iloc[top]
    return cell.name[1].astype(int), cell.name[0].astype(int)

def get_median_cell(df, metric='change_modulation',top=0):
    temp = df.groupby(['cell','ophys_experiment_id']).mean().sort_values(by=metric)
    cell = temp.iloc[int(np.floor(len(temp)/2))+top]
    return cell.name[1].astype(int), cell.name[0].astype(int)

def add_num_blocks(df):
    num_blocks = df.groupby(['ophys_experiment_id','cell']).size().to_frame(name='num_blocks').reset_index()
    new_df = df.merge(num_blocks, how = 'inner',on=['ophys_experiment_id','cell'])
    return new_df


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



