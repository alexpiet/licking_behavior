import numpy as np
import pandas as pd
import psy_tools as ps
import psy_general_tools as pgt
import matplotlib.pyplot as plt
plt.ion()



def get_train_summary(output_dir='/home/alex.piet/codebase/behavior/model_output/'):
    '''
        Loads a dataframe from file with model information. 
        This csv is created in psy_output_tools.build_training_summary_table()
    '''
    train_summary = pd.read_csv(output_dir+'_training_summary_table.csv')
    return train_summary



def plot_mouse_strategy_correlation(train_summary,group_label='',metric='task_dropout_index'):
    '''
        Plots each mouse's difference in strategy from its final strategy. 
    '''

    # Get the average ophys value for each mouse averaged across imaging days (negative pre_ophys_numbers)
    mouse_summary = train_summary.pivot(index='donor_id',columns='pre_ophys_number',values=[metric])
    mouse_summary['ophys_index'] = mouse_summary[metric][0]
    mouse_summary = mouse_summary.copy()

    # Plot each mouse's trajectory    
    plt.figure(figsize=(10,5))
    plt.axvspan(0,6,color='k',alpha=.1)
    plt.axhline(0, color='k',linestyle='--',alpha=0.5)
    xvals = -np.sort(train_summary.pre_ophys_number.unique())
    for dex, mouse in enumerate(range(0,len(mouse_summary))):
        plt.plot(xvals, mouse_summary[metric][:].iloc[dex].values- mouse_summary['ophys_index'].iloc[dex],'o-',alpha=.1)

    # Plot the mean trajectory
    means = []
    for dex,val in enumerate(np.sort(train_summary.pre_ophys_number.unique())):
        means.append(np.nanmean(mouse_summary[metric][val].values - mouse_summary['ophys_index'].values))
        plt.plot(-val, np.nanmean(mouse_summary[metric][val].values - mouse_summary['ophys_index'].values),'rx')
    plt.plot(xvals, means, 'r-',linewidth=2) 
    plt.xlabel('Sessios before Ophys Stage 1', fontsize=16)  
    plt.xlim(right=6)

    # Save and cleanup
    if metric is not 'task_dropout_index':
        plt.ylabel('Diff in '+metric,fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/mouse_strategy_correlation'+group_label+'_'+metric+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/mouse_strategy_correlation'+group_label+'_'+metric+'.png')
    else: 
        plt.ylim(-25,25)  
        plt.ylabel('Diff in Strategy Index',fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/mouse_strategy_correlation'+group_label+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/mouse_strategy_correlation'+group_label+'.png')



def plot_strategy_correlation(train_summary,min_sessions=10,group_label='',metric='task_dropout_index',corr_method = 'pearson'):
    '''
        Makes a plot that computes the correlation of each mouse's strategy index across training days. 
        For each training day it computes the correlation with the average values on imaging days.  

        min_sessions is the minimum number of sessions for each day to compute the correlation
    '''
    
    # Get the average ophys value for each mouse averaged across imaging days (negative pre_ophys_numbers)
    mouse_summary = train_summary.pivot(index='donor_id',columns='pre_ophys_number',values=[metric])
    mouse_summary['ophys_index'] = mouse_summary[metric][0]

    # Build Plot
    plt.figure(figsize=(10,5))
    plt.axvspan(0,6,color='k',alpha=.1)
    plt.axhline(0, color='k',linestyle='--',alpha=0.5)
    plt.xlabel('Sessions before Ophys Stage 1',fontsize=16)
    plt.xlim(right=6)  

    # Iterate through training days
    for dex,val in enumerate(train_summary.pre_ophys_number.unique()): 
        if len(mouse_summary[metric][val].unique())> min_sessions:
            plt.plot(-val, mouse_summary['ophys_index'].corr(mouse_summary[metric][val],method=corr_method),'ko')
      
    # Clean up and save
    if metric is not 'task_dropout_index':
        plt.ylabel(metric+' Correlation ('+corr_method+')',fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/strategy_correlation'+group_label+'_'+metric+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/strategy_correlation'+group_label+'_'+metric+'.png')
    else: 
        plt.ylabel('Strategy Index Correlation ('+corr_method+')',fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/strategy_correlation'+group_label+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/strategy_correlation'+group_label+'.png')



def plot_training(train_summary, mark_start=False,group_label='',metric='task_dropout_index'):
    '''
        Plots the strategy index for each mouse by session day, colored by the final strategy index
    
        train_summary is found in  _training_summary_table.csv

        if mark_start, then marks the start of training for each mouse
    '''
    
    # Get list of mice with imaging data
    donor_ids = train_summary.query('imaging').donor_id.unique()

    # Make figure
    plt.figure(figsize=(10,5))
    plt.axhline(0,color='k',linestyle='--',alpha=0.5) 
    plt.axvspan(0,6,color='k',alpha=.1)
    x = []
    y = []
    c = []

    # Iterate across mice
    for dex, donor_id in enumerate(donor_ids): 
        # Filter for this mouse
        mouse_table = train_summary.query('donor_id == @donor_id')
        vals = mouse_table[metric].values
        xvals = -mouse_table.pre_ophys_number.values
    
        # if we have sessions
        if len(vals) > 0:
            plt.plot(xvals, vals,'k-',alpha=.05)
            x = x + list(xvals)
            y = y + list(vals)
            c = c + list(np.ones(np.size(vals))*mouse_table.query('imaging')[metric].mean())
            if mark_start:
                plt.plot(xvals[0],vals[0],'kx',alpha=0.5)

    # plot all the data with a common color map
    scat = plt.gca().scatter(x, y, s=80,c =c, cmap='plasma',alpha=0.5)
    plt.xlabel('Sessions before Ophys Stage 1',fontsize=16)
    plt.xlim(right=6)

    # Save and clean up
    if metric is not 'task_dropout_index':
        plt.ylabel(metric,fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_session_number'+group_label+'_'+metric+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_session_number'+group_label+'_'+metric+'.png')  
    else:
        plt.ylabel('Strategy Index',fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_session_number'+group_label+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_session_number'+group_label+'.png')



def plot_training_by_stage(train_summary,group_label='',metric='task_dropout_index',corr_method='pearson'):
    '''
        Plot the strategy index for the first and last session of each training stage, colored by the final strategy index
        train_summary is found in  _training_summary_table.csv

        dev function, plots by stage
    '''

    # Organize mouse data
    donor_ids = train_summary.query('imaging').donor_id.unique()
    mouse_summary = train_summary.pivot(index='donor_id',columns='pre_ophys_number',values=[metric])
    mouse_summary['ophys_index'] = mouse_summary[metric][0]

    # Make figure
    plt.figure(figsize=(10,5))
    plt.axhline(0,color='k',linestyle='--',alpha=0.5) 
    x = []
    y = []
    c = []
    corr_data = []
   
    # Build Mouse first/last data 
    for dex, donor_id in enumerate(donor_ids):
        mouse_table = train_summary.query('donor_id == @donor_id')

        idex = [
        mouse_table.query('(not ophys) & (stage == "3")').first_valid_index(),
        mouse_table.query('(not ophys) & (stage == "3")').last_valid_index(),
        mouse_table.query('(not ophys) & (stage == "4")').first_valid_index(),
        mouse_table.query('(not ophys) & (stage == "4")').last_valid_index(),
        mouse_table.query('(not ophys) & (stage == "5")').first_valid_index(),
        mouse_table.query('(not ophys) & (stage == "5")').last_valid_index(),
        mouse_table.query('(ophys) & (stage == "0")').first_valid_index(),
        mouse_table.query('(ophys) & (stage == "0")').last_valid_index(),
        mouse_table.query('(ophys) & (stage == "1")').first_valid_index(),
        mouse_table.query('(ophys) & (stage == "3")').first_valid_index(),
        mouse_table.query('(ophys) & (stage == "4")').first_valid_index(),
        mouse_table.query('(ophys) & (stage == "6")').first_valid_index()]
    
        vals = []
        for dex2, val in enumerate(idex):
            if val is not None:
                vals.append(mouse_table.loc[val][metric])
            else:
                vals.append(np.nan)
        xvals = [-3,-2.75,-2,-1.75,-1,-0.75,0,0.25,1,2,3,4]
        plt.plot(xvals, vals,'k-',alpha=.05)
        x = x + xvals
        y = y + vals
        c = c + list(np.ones(np.size(vals))*mouse_table.query('imaging')[metric].mean())
        corr_data.append(vals)

    # Plot the stage information 
    scat = plt.gca().scatter(x, y, s=80,c =c, cmap='plasma',alpha=0.5)
    plt.xlabel('Stage',fontsize=16)
    plt.xticks([-3,-2,-1,0,1,2,3,4], ['T3','T4','T5','Hab', 'Ophys1','Ophys3','Ophys4','Ophys6'],fontsize=14)
    plt.yticks(fontsize=14)
   
    # Plot and save 
    if metric is not 'task_dropout_index':
        plt.ylabel(metric,fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/first_last_by_stage'+group_label+'_'+metric+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/first_last_by_stage'+group_label+'_'+metric+'.png')
    else:
        plt.ylabel('Strategy Index',fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/first_last_by_stage'+group_label+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/first_last_by_stage'+group_label+'.png')

    # Plot the correlation data
    corr_data = np.vstack(corr_data)
    plot_strategy_correlation_by_stage(corr_data,group_label=group_label, metric=metric,corr_method=corr_method)



def plot_strategy_correlation_by_stage(corr_data,group_label='',metric='task_dropout_index',ref_index=8,corr_method='pearson'):
    mouse_summary = pd.DataFrame(data = corr_data)

    # Build Plot
    plt.figure(figsize=(10,5))
    plt.axvspan(1,4.5,color='k',alpha=.1)
    plt.axhline(0, color='k',linestyle='--',alpha=0.5)
    xvals = [-3,-2.75,-2,-1.75,-1,-0.75,0,0.25,1,2,3,4]
    plt.xlabel('Stage',fontsize=16)
    plt.xticks([-2.875,-1.875,-0.875,0.125,1,2,3,4], ['T3','T4','T5','Hab', 'Ophys1','Ophys3','Ophys4','Ophys6'],fontsize=14)
    plt.yticks(fontsize=14)

    # Iterate through training days
    for dex,val in enumerate(mouse_summary.keys()):
        plt.plot(xvals[dex], mouse_summary[val].corr(mouse_summary[ref_index],method=corr_method),'ko')
    
    # Clean up and save
    if metric is not 'task_dropout_index':
        plt.ylabel(metric+' Correlation ('+corr_method+')',fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/first_last_by_stage_strategy_correlation'+group_label+'_'+metric+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/first_last_by_stage_strategy_correlation'+group_label+'_'+metric+'.png')
    else: 
        plt.ylabel('Strategy Index Correlation ('+corr_method+')',fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/first_last_by_stage_strategy_correlation'+group_label+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/first_last_by_stage_strategy_correlation'+group_label+'.png')



def plot_training_dropout(train_summary,group_label='',metric='task_dropout_index'):
    '''
        train_summary is found in  _training_summary_table.csv

        dev function, plots by stage
    '''
    donor_ids = train_summary.query('imaging').donor_id.unique()

    plt.figure(figsize=(10,5))
    plt.axhline(0,color='k',linestyle='--',alpha=0.5) 
    x = []
    y = []
    c = []
    for dex, donor_id in enumerate(donor_ids):
        mouse_table = train_summary.query('donor_id == @donor_id')
        vals = [mouse_table.query('(not ophys) & (stage == "3")')[metric].mean(),
        mouse_table.query('(not ophys) & (stage == "4")')[metric].mean(),
        mouse_table.query('(not ophys) & (stage == "5")')[metric].mean(),
        mouse_table.query('(ophys) & (stage == "0")')[metric].mean(),
        mouse_table.query('(ophys) & (stage == "1")')[metric].mean(),
        mouse_table.query('(ophys) & (stage == "3")')[metric].mean(),
        mouse_table.query('(ophys) & (stage == "4")')[metric].mean(),
        mouse_table.query('(ophys) & (stage == "6")')[metric].mean()]
        xvals = [-3,-2,-1,0,1,2,3,4]
        plt.plot(xvals, vals,'k-',alpha=.05)
        x = x + xvals
        y = y + vals
        c = c + list(np.ones(np.size(vals))*mouse_table.query('imaging')[metric].mean())

    scat = plt.gca().scatter(x, y, s=80,c =c, cmap='plasma',alpha=0.5)

    vals = [train_summary.query('(not ophys) & (stage == "3")')[metric].mean(),
    train_summary.query('(not ophys) & (stage == "4")')[metric].mean(),
    train_summary.query('(not ophys) & (stage == "5")')[metric].mean(),
    train_summary.query('(ophys) & (stage == "0")')[metric].mean(),
    train_summary.query('(ophys) & (stage == "1")')[metric].mean(),
    train_summary.query('(ophys) & (stage == "3")')[metric].mean(),
    train_summary.query('(ophys) & (stage == "4")')[metric].mean(),
    train_summary.query('(ophys) & (stage == "6")')[metric].mean()]
    plt.plot(xvals, vals, 'k-',linewidth=2)

    plt.xlabel('Stage',fontsize=16)
    plt.xticks(xvals, ['T3','T4','T5','Hab', 'Ophys1','Ophys3','Ophys4','Ophys6'],fontsize=14)
    plt.yticks(fontsize=14)
    
    if metric is not 'task_dropout_index':
        plt.ylabel(metric,fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_stage'+group_label+'_'+metric+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_stage'+group_label+'_'+metric+'.png')
    else:
        plt.ylabel('Strategy Index',fontsize=16)
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_stage'+group_label+'.svg')
        plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/summary_by_stage'+group_label+'.png')



def plot_training_roc(train_summary,group_label=''):
    '''
        train_summary is found in  _training_summary_table.csv
        
        plots AU-ROC as a function of training stage for each mouse
    '''
    donor_ids = train_summary.query('imaging').donor_id.unique()

    plt.figure(figsize=(10,5))
    plt.axhline(train_summary.session_roc.mean(),color='k',linestyle='--',alpha=0.5) 
    
    x = []
    y = []
    c = []
    for dex, donor_id in enumerate(donor_ids):
        mouse_table = train_summary.query('donor_id == @donor_id')
        vals = [mouse_table.query('(not ophys) & (stage == "3")').session_roc.mean(),
        mouse_table.query('(not ophys) & (stage == "4")').session_roc.mean(),
        mouse_table.query('(not ophys) & (stage == "5")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "0")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "1")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "3")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "4")').session_roc.mean(),
        mouse_table.query('(ophys) & (stage == "6")').session_roc.mean()]
        xvals = [-3,-2,-1,0,1,2,3,4]
        plt.plot(xvals, vals,'k-',alpha=.1)
        x = x + xvals
        y = y + vals
        c = c + vals 

    scat = plt.gca().scatter(x, y, s=80,alpha=0.5)

    vals = [train_summary.query('(not ophys) & (stage == "3")').session_roc.mean(),
    train_summary.query('(not ophys) & (stage == "4")').session_roc.mean(),
    train_summary.query('(not ophys) & (stage == "5")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "0")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "1")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "3")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "4")').session_roc.mean(),
    train_summary.query('(ophys) & (stage == "6")').session_roc.mean()]
    plt.plot(xvals, vals, 'm-',linewidth=2)

    plt.ylabel('Session ROC',fontsize=16)
    plt.xlabel('Stage',fontsize=16)
    plt.xticks(xvals, ['T3','T4','T5','Hab', 'Ophys1','Ophys3','Ophys4','Ophys6'],fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylim(0.6,1)

    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/roc_by_stage'+group_label+'.svg')
    plt.savefig('/home/alex.piet/codebase/behavior/training_analysis/roc_by_stage'+group_label+'.png')


