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

def dev_analyze_manifest(manifest):
    plt.figure()
    manifest.groupby(driver_line)
    # Average by cre-line
    # Average over all?
    # make matrix of coefficients?


def dev_run_manifest(manifest):
    manifest['model_mean'] = np.nan
    manifest['model_r2'] =  [[]] * len(manifest)
    manifest['model_coefs'] =  [[]] * len(manifest)
    for index,row in manifest.iterrows():
        id = row.ophys_experiment_id
        print(str(id))
        try:
            m = run_session(id,plot_this=False,verbose=False)
            manifest.at[index,'model_mean'] = np.mean(m[0])
            manifest.at[index,'model_r2'] = m[0]
            manifest.at[index,'model_coefs'] = m[1]
        except:
            pass
    return manifest

def dev_run_session(id,plot_this=True,verbose=False):
    fit = ps.load_fit(id)   
    session = ps.get_data(id)
    pm.get_metrics(session)
    cells = session.flash_response_df['cell_specimen_id'].unique()
    r2 = []
    coefs = []
    r22 = []
    coefs2 = []
    r23 = []
    coefs3 = []
    r24 = []
    coefs4 = []
    r25 = []
    coefs5 = []
    r26 = []
    coefs6 = []
    for cell in cells:
        score,coef,reg,scores = run_cell_ridge(cell,session,fit,verbose=verbose,   use_metrics=True,  use_model=True, use_task=True)
        score2,coef2,reg2,scores2 = run_cell_ridge(cell,session,fit,verbose=verbose,use_metrics=False, use_model=True, use_task=True)
        score3,coef3,reg3,scores3 = run_cell_ridge(cell,session,fit,verbose=verbose,use_metrics=True,  use_model=False,use_task=True)
        score4,coef4,reg4,scores4 = run_cell_ridge(cell,session,fit,verbose=verbose,use_metrics=True,  use_model=False,use_task=False)
        score5,coef5,reg5,scores5 = run_cell_ridge(cell,session,fit,verbose=verbose,use_metrics=False, use_model=False,use_task=True)
        score6,coef6,reg6,scores6 = run_cell_ridge(cell,session,fit,verbose=verbose,use_metrics=False, use_model=True, use_task=False)
        r2.append(score)
        coefs.append(coef)
        r22.append(score2)
        coefs2.append(coef2)
        r23.append(score3)
        coefs3.append(coef3)
        r24.append(score4)
        coefs4.append(coef4)
        r25.append(score5)
        coefs5.append(coef5)
        r26.append(score6)
        coefs6.append(coef6)
    if plot_this:
        fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(12,4))
        ax[0].plot(np.sort(r2),'b')
        ax[0].plot(np.sort(r22),'r')
        ax[0].plot(np.sort(r23),'g')
        ax[0].plot(np.sort(r24),'k')
        ax[0].plot(np.sort(r25),'m')
        ax[0].plot(np.sort(r26),'c')
        ax[0].set_title(str(id))
        ax[0].set_ylim(bottom=0)
        ax[0].set_ylabel('r^2')
        ax[0].set_xlabel('Cell (sorted)')
        #labels = ['bias','task0','omissions1','timing','bout_rate','reward_rate','change_regressor','reward_regressor','mean_running_speed']
        labels = ['bias','task0','omissions1','timing','bout_rate','reward_rate','change_regressor']
        ax[1].plot(range(0,7),np.mean(np.abs(coefs),0),'b')
        ax[1].plot(list(range(0,4))+[6],np.mean(np.abs(coefs2),0),'r')
        ax[1].plot(range(4,7),np.mean(np.abs(coefs3),0),'g')
        ax[1].plot(range(4,6),np.mean(np.abs(coefs4),0),'k')
        ax[1].plot(range(6,7),np.mean(np.abs(coefs5),0),'m')
        ax[1].plot(range(0,4),np.mean(np.abs(coefs6),0),'c')
        ax[1].set_xticks(range(0,len(labels)))
        ax[1].set_xticklabels(labels,rotation=90)
        ax[1].set_ylabel('Avg. Abs. Weight')
        temp = np.vstack([r2,r22,r23,r24,r25,r26])
        sem  = np.std(temp,1)/np.sqrt(len(r2))
        ax[2].plot(range(0,6),temp,'ko',alpha=0.1)
        #ax[2].plot(range(0,6),np.mean(temp,1),'ro')
        colors=['b','r','g','k','m','c']
        for i in range(0,len(sem)):
            ax[2].plot([i],np.mean(temp,1)[i],'o',color=colors[i])
            ax[2].plot([i,i],[np.mean(temp,1)[i]-sem[i], np.mean(temp,1)[i]+sem[i]],'-',color=colors[i])
        ax[2].set_ylabel('r^2')
        ax[2].set_xticks(range(0,7))
        ax[2].set_xticklabels(['Full','No Metrics','No Model','Just Metrics','Just Task','Just Model'],rotation=90)
        ax[2].set_xlim(-1,6)
        plt.tight_layout()
    return r2,coefs

def dev_get_cell_flash_df(cell_id,session,fit,all_stim=False):
    cell_flash_df = get_cell_df(cell_id,session)
    cell_flash_df = add_metrics(cell_flash_df,session)
    cell_flash_df = remove_licking_bout(cell_flash_df,fit)
    cell_flash_df = add_model_weights(cell_flash_df,fit)
    if not all_stim:
        cell_flash_df = cell_flash_df[cell_flash_df['pref_stim']] 
    return cell_flash_df

def dev_run_cell_ridge(cell_id,session,fit,verbose=False,use_metrics=True,use_model=True,use_task=True,plot_cell=False,use_full=False):
    cell_flash_df = get_cell_flash_df(cell_id,session,fit)
    regressors = []
    if use_model:
        regressors +=['bias','task0','omissions1','timing']
    if use_metrics:
        regressors += ['bout_rate','reward_rate']
    if use_task:
        #regressors += ['change_regressor','reward_regressor','mean_running_speed']
        regressors += ['change_regressor']
    if use_full:
        regressors += ['full']
    x,y = build_design_matrix(cell_flash_df,regressors)
    #y = cell_flash_df.mean_response.values
    reg =linear_model.RidgeCV(alphas=np.logspace(-6,6,13),store_cv_values=True)
    reg.fit(x,y)
    scores= model_selection.cross_validate(reg,x,y=y,cv=10,return_train_score=True) 
    if plot_cell & (len(regressors) == 1):
        plt.figure()
        plt.plot(x,y,'ko')
    if verbose:
        print(str(cell_id) +" : " +str(np.mean(scores['train_score']))+" : "+str(np.mean(scores['test_score'])) )
    return np.mean(scores['test_score']),reg.coef_,reg,scores

def dev_run_cell_ols(cell_id,session,fit,verbose=True,use_task_state=True):
    cell_flash_df = get_cell_df(cell_id,session)
    cell_flash_df = add_metrics(cell_flash_df,session)
    cell_flash_df = remove_licking_bout(cell_flash_df,fit)
    cell_flash_df = add_model_weights(cell_flash_df,fit,use_task_state=use_task_state)
    cell_flash_df = cell_flash_df[cell_flash_df['pref_stim']] 
    if use_task_state:
        x,y = build_design_matrix(cell_flash_df,list(sorted(fit['weights'].keys()))+['bout_rate','reward_rate'])
    else:
        x,y = build_design_matrix(cell_flash_df,list(sorted(fit['weights'].keys())))

    reg =linear_model.LinearRegression()
    reg.fit(x,y)
    scores= model_selection.cross_validate(reg,x,y=y,cv=20,return_train_score=True) 
    if verbose:
        print(str(cell_id) +" : " +str(np.mean(scores['train_score']))+" : "+str(np.mean(scores['test_score'])) )
    return reg.score(x,y),reg.coef_,scores


def dev_add_metrics(cell_flash_df,session):
    cell_flash_df = cell_flash_df.assign(bout_rate = session.stimulus_presentations['bout_rate'].values)
    cell_flash_df = cell_flash_df.assign(reward_rate = session.stimulus_presentations['reward_rate'].values)
    return cell_flash_df

def dev_remove_licking_bout(cell_flash_df,fit):
    cell_flash_df = cell_flash_df.assign(in_bout = fit['psydata']['full_df']['in_bout'].values)
    cell_flash_df = cell_flash_df[cell_flash_df['in_bout'] != 1]
    return cell_flash_df

def dev_add_model_weights(cell_flash_df,fit):
    model_df = pd.DataFrame(fit['wMode'].T,columns=list(sorted(fit['weights'].keys())))
    model_df = model_df.reset_index()
    
    #modulate_by_task_state
    for key in model_df.columns:
        if (key == 'bias') | (key == 'index'):
            pass
        else:
            model_df[key] = model_df[key].values*fit['psydata']['df'][key].values
 
    cell_flash_df = cell_flash_df.reset_index()
    cell_flash_df = pd.concat([cell_flash_df,model_df],axis=1)

    # Grouping timing indicies together
    cell_flash_df['timing'] =   cell_flash_df['timing2'] + cell_flash_df['timing3'] + cell_flash_df['timing4'] + cell_flash_df['timing5'] + cell_flash_df['timing6'] + cell_flash_df['timing7'] + cell_flash_df['timing8'] + cell_flash_df['timing9'] + cell_flash_df['timing10'] 
    cell_flash_df['full'] = cell_flash_df['bias'] + cell_flash_df['omissions1'] + cell_flash_df['task0'] + cell_flash_df['timing'] 
    
    
    cell_flash_df['change_regressor'] = cell_flash_df['change'].astype(int)
    cell_flash_df['reward_regressor'] = (cell_flash_df['rewards'].str.len() > 0).astype(int)
    return cell_flash_df

def dev_build_design_matrix(cell_flash_df,regressor_list,not_change=True):
    if not_change:
        temp = cell_flash_df[~cell_flash_df['change']]
    else:
        temp = cell_flash_df
    design_matrix = temp[regressor_list].to_numpy()
    y = temp.mean_response.values
    return design_matrix,y



def dev_plot_summary(id):
    session = ps.get_data(id)
    fit = ps.load_fit(id)
    pm.get_metrics(session)
    cells = session.flash_response_df['cell_specimen_id'].unique()
    ff = ps.plot_fit(id)
    cell_flash_df = get_cell_flash_df(cells[0],session,fit,all_stim=True)
    # Get average df/f over all cells
    flashes = session.flash_response_df['flash_id'].unique()
    fr = session.flash_response_df
    average_dffs=[]
    task = []
    bias = []
    for f in flashes:
        if not session.stimulus_presentations.loc[f]['change']:
            if np.sum(cell_flash_df['flash_id'] == f)> 0:
                average_dffs.append(fr[(fr['flash_id'] == f)& (fr['pref_stim'])]['mean_response'].mean())
                #task.append(cell_flash_df[cell_flash_df['flash_id']==f]['task0'].values[0])
                bias.append(cell_flash_df[cell_flash_df['flash_id']==f]['bias'].values[0])
    bias = np.array(bias)[~np.isnan(average_dffs)]
    #task = np.array(task)[~np.isnan(average_dffs)]
    average_dffs = np.array(average_dffs)[~np.isnan(average_dffs)]
    # Plot summary figure
    reg =linear_model.LinearRegression()
    reg.fit(np.array(bias).reshape(-1,1),np.array(average_dffs))
    bp = np.sort(np.array(bias)).reshape(-1,1)
    #tp = np.sort(np.array(task)).reshape(-1,1)
    plt.figure()
    plt.plot(bias,average_dffs,'ko')
    plt.ylabel('Average df/f')
    plt.xlabel('Bias')
    plt.plot(bp,reg.predict(bp),'r')
    plt.title(id)
    #plt.figure()
    #plt.plot(task,average_dffs,'ko')
    #plt.ylabel('Average df/f')
    #plt.xlabel('Task')
    #plt.plot(tp,reg.predict(tp),'r')
    #plt.title(id)
    return reg.coef_, reg.score(np.array(bias).reshape(-1,1),np.array(average_dffs))

def dev_get_session_change_modulation(id):
    fit = ps.load_fit(id)   
    session = ps.get_data(id)
    pm.get_metrics(session)
    cells = session.flash_response_df['cell_specimen_id'].unique()
    cms = []
    average_cms = []
    fulls = []
    tasks = []
    biass = []
    hits = []
    for cell in cells:
        cm,average_cm,full,task,bias,hit = get_cell_change_modulation(cell,session,fit)
        cms.append(cm)
        average_cms.append(average_cm)
        fulls.append(full)
        tasks.append(task)
        biass.append(bias)
        hits.append(hit)
    return cms,average_cms,fulls,tasks,biass,hits

def dev_get_cell_change_modulation(cell,session,fit):
    cell_flash_df = get_cell_flash_df(cell,session,fit)    
    #cell_flash_df = get_cell_df(cell,session)
    #cell_flash_df = cell_flash_df[cell_flash_df['pref_stim']]
    cms = []
    fulls = []
    tasks = []
    biass = []
    change = []
    post = []
    hit = []
    if is_cell_significant(cell_flash_df):
        blocks = cell_flash_df['block_index'].unique()
        for block in blocks:        
            block_df = cell_flash_df[cell_flash_df['block_index'] == block] 
            if len(block_df) > 10:
                #cms.append((block_df.iloc[0]['mean_response'] - block_df.iloc[9]['mean_response'])/ (np.abs(block_df.iloc[0]['mean_response']) + np.abs(block_df.iloc[9]['mean_response'])))
                cms.append((block_df.iloc[0]['mean_response'] - block_df.iloc[9]['mean_response'])/ (block_df.iloc[0]['mean_response'] + block_df.iloc[9]['mean_response']))
                change.append(block_df.iloc[0]['mean_response'])
                post.append(block_df.iloc[9]['mean_response'])
                fulls.append(block_df.iloc[0]['full'])
                tasks.append(block_df.iloc[0]['task0'])
                biass.append(block_df.iloc[0]['bias'])
                hit.append(flash_hit(block_df.iloc[0]['flash_id'],session))
    #if len(cms) > 0:
    #    plot_cell_change_modulation(cms,fulls,tasks,biass)
        average_cm = (np.mean(change) - np.mean(post))/(np.mean(np.abs(change))+np.mean(np.abs(post)))
    else:
        average_cm = np.nan
    return cms,average_cm, fulls,tasks,biass,hit

def dev_plot_cell_change_modulation(cm,full,task,bias):
    fig,ax = plt.subplots(nrows=3,ncols=1)
    ax[0].plot(full, cm,'ko')
    ax[1].plot(task, cm,'ko')
    ax[2].plot(bias, cm,'ko')

def dev_is_cell_significant(cell_flash_df):
    return (np.sum(cell_flash_df['p_value'] < 0.05)/len(cell_flash_df)) > 0.25


def dev_streamline_get_session_change_modulation(id):
    #fit = ps.load_fit(id)  
    #dropout = np.empty((len(fit['models']),))
    #for i in range(0,len(fit['models'])):
    #    dropout[i] = (1-fit['models'][i][1]/fit['models'][0][1])*100
    #dex = -(dropout[2] - dropout[18])
    session = ps.get_data(id)
    #stage = session.metadata['stage']
    cells = session.flash_response_df['cell_specimen_id'].unique()
    cms = []
    for cell in cells:
        cm = streamline_get_cell_change_modulation(cell,session)
        cms.append(cm)
    return cms#,dex,stage

def dev_streamline_get_cell_change_modulation(cell,session):
    cell_flash_df = get_cell_df(cell,session)
    cell_flash_df = cell_flash_df[cell_flash_df['pref_stim']]
    cms = []
    if is_cell_significant(cell_flash_df):
        blocks = cell_flash_df['block_index'].unique()
        for block in blocks:        
            block_df = cell_flash_df[cell_flash_df['block_index'] == block] 
            if len(block_df) > 10:
                cms.append((block_df.iloc[0]['mean_response'] - block_df.iloc[9]['mean_response'])/ (block_df.iloc[0]['mean_response'] + block_df.iloc[9]['mean_response']))
    else:
        average_cm = np.nan
    return cms

def dev_flash_hit(flash_id,session):
    return len(session.stimulus_presentations.loc[flash_id]['rewards']) > 0


##### Dev above here

def get_cell_df(cell_id, session):
    return session.flash_response_df[session.flash_response_df['cell_specimen_id'] == cell_id]

def test_cell_reliability(cell_flash_df, pval=0.05, percent=0.25):
    return (np.sum(cell_flash_df['p_value'] < pval)/len(cell_flash_df)) > percent

def cell_change_modulation(cell, session,remove_unreliable=True, use_mean_response=False):
    '''
        pref_stim ?
        reliable cells ?
        licking bouts?
        all flashes, or just trials?
        remove outsides (-1,1)? 
    '''
    cell_flash_df = get_cell_df(cell,session)
    cell_flash_df = cell_flash_df[cell_flash_df['pref_stim']]
    cms = []

    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],'change_modulation':[],'flash_id':[],'change_modulation_base':[]})


    if remove_unreliable:
        percent = 0.25
    else:
        percent = 0
    if test_cell_reliability(cell_flash_df,percent=percent):
        blocks = cell_flash_df['block_index'].unique()
        for block in blocks:        
            block_df = cell_flash_df[cell_flash_df['block_index'] == block] 
            if len(block_df) > 10:
                this_cm = (block_df.iloc[0]['mean_response'] - block_df.iloc[9]['mean_response'])/ (block_df.iloc[0]['mean_response'] + block_df.iloc[9]['mean_response'])
                this_change = block_df.iloc[0]['mean_response'] - block_df.iloc[0]['baseline_response']
                this_non    = block_df.iloc[9]['mean_response'] - block_df.iloc[9]['baseline_response']
                this_cm_base =(this_change - this_non)/(this_change + this_non)
                if (this_cm > -1) & (this_cm < 1) &(this_cm_base > -1) & (this_cm_base < 1):
                    cms.append(this_cm)
                    d = {'ophys_experiment_id':session.metadata['ophys_experiment_id'],'stage':session.metadata['stage'],'cell':cell,'imaging_depth':session.metadata['imaging_depth'],'change_modulation':this_cm,'flash_id':block_df.iloc[0]['flash_id'],'change_modulation_base':this_cm_base}
                    df = df.append(d,ignore_index=True)

    return df, cms

def session_change_modulation(id,remove_unreliable=True):
    session = ps.get_data(id)
    cells = session.flash_response_df['cell_specimen_id'].unique()
    all_cms = []
    mean_cms = []
    var_cms = []
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],'change_modulation':[],'flash_id':[],'change_modulation_base':[]})
    for cell in cells:
        cell_df,cm = cell_change_modulation(cell,session,remove_unreliable=remove_unreliable)
        if len(cm) > 0:
            all_cms.append(cm)
            mean_cms.append(np.mean(cm))
            var_cms.append(np.var(cm))
            df = df.append(cell_df,ignore_index=True)
    session_mean = np.mean(mean_cms)
    session_var = np.var(mean_cms)
    return df, all_cms,mean_cms,var_cms, session_mean, session_var

def manifest_change_modulation(ids,remove_unreliable=True):
    all_cms = []
    mean_cms = []
    var_cms = []
    session_means = []
    session_vars = []
    df = pd.DataFrame(data={'ophys_experiment_id':[],'stage':[],'cell':[],'imaging_depth':[],'change_modulation':[],'flash_id':[],'change_modulation_base':[]})
    for id in tqdm(ids):
        session_df, cell_cms,cell_mean_cms,cell_var_cms, session_mean,session_var = session_change_modulation(id,remove_unreliable=remove_unreliable)
        all_cms.append(cell_cms)
        mean_cms.append(cell_mean_cms)
        var_cms.append(cell_var_cms)
        session_means.append(session_mean)
        session_vars.append(session_var)
        df = df.append(session_df,ignore_index=True)
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
    plt.figure()
    for index,dist in enumerate(dists):
        counts,edges = np.histogram(dist,bins[index])
        centers = edges[0:-1] + np.diff(edges)/2
        plt.bar(centers, counts/np.sum(counts)/np.diff(edges)[0],width=np.diff(edges),color=colors[index],alpha=alpha[index], label=labels[index])
    plt.legend()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(-1,1)
    plt.gca().axvline(0,linestyle='--',color='k',alpha=0.3)

def compare_means(groups,colors,labels,ylim,axislabels):
    plt.figure()
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

def compare_groups_df(dfs,labels,metric='change_modulation', xlabel="Change Modulation",alpha=0.5, nbins=[5,50,50],savename=None,nboots=1000):
    if type(savename) is not type(None):
        filepath = '/home/alex.piet/codebase/behavior/doc/figures/change_modulation_figures/'+savename
        if metric is not 'change_modulation':
            filepath = filepath + "_"+metric
    else:
        filepath = None

    numdfs = len(dfs)
    colors = sns.color_palette(n_colors=len(dfs))

    for index, df in enumerate(dfs):
        plot_manifest_change_modulation_df(df,plot_cells=False,titlestr=labels[index],filepath=filepath,metric=metric)

    compare_session_dist_df(dfs, np.tile(nbins[0],numdfs),colors,labels,np.tile(alpha,numdfs),xlabel="Session Avg."+xlabel,ylabel="Prob",filepath=filepath,metric=metric)
    compare_cell_dist_df(dfs,    np.tile(nbins[1],numdfs),colors,labels,np.tile(alpha,numdfs),xlabel="Cell Avg. "+xlabel,ylabel="Prob",filepath=filepath,metric=metric) 
    compare_flash_dist_df(dfs,   np.tile(nbins[2],numdfs),colors,labels,np.tile(alpha,numdfs),xlabel="Flash "+xlabel,ylabel="Prob",filepath=filepath,metric=metric) 
    compare_means_df(dfs, labels,filepath=filepath,metric=metric,nboots=nboots)

def annotate_stage(df):
    df['image_set'] = [x[15] for x in df['stage'].values]
    df['active'] = [ (x[6] in ['1','3','4','6']) for x in df['stage'].values]
    df['stage_num'] = [x[6] for x in df['stage'].values]
    return df

def compare_means_df(dfs,df_labels,metric='change_modulation',ylabel='Change Modulation',labels=['Flash','Cell','Session'],ylim=[0,.2],filepath=None,titlestr="",nboots=1000):
    plt.figure()
    colors = sns.color_palette(n_colors=len(dfs))

    df_flash_boots = []
    df_cell_boots = []
    for index, df in enumerate(dfs):
        dists = [df[metric].values,  df.groupby(['ophys_experiment_id','cell']).mean()[metric].values,df.groupby(['ophys_experiment_id']).mean()[metric].values]
        if nboots > 0:
            flash_boots = bootstrap_df(df,nboots,metric=metric)
            df_flash_boots.append(flash_boots[2])
            plt.plot(0.1,flash_boots[0],'o',color=colors[index],alpha=0.5)
            plt.plot([0.1,0.1],[flash_boots[0]-flash_boots[1], flash_boots[0]+flash_boots[1]],'-',color=colors[index],alpha=0.5)

            cell_boots = bootstrap_df(df.groupby(['ophys_experiment_id','cell']).mean().reset_index(),nboots,levels=['root','ophys_experiment_id'],metric=metric)
            df_cell_boots.append(cell_boots[2])
            plt.plot(1.1,cell_boots[0],'o',color=colors[index],alpha=0.5)
            plt.plot([1.1,1.1],[cell_boots[0]-cell_boots[1], cell_boots[0]+cell_boots[1]],'-',color=colors[index],alpha=0.5)

        for ddex,dist in enumerate(dists):
            if ddex == 0:
                plt.plot(ddex, np.mean(dist),'o',color=colors[index],label=df_labels[index])
            else:
                plt.plot(ddex, np.mean(dist),'o',color=colors[index])
            plt.plot([ddex,ddex],[np.mean(dist)-np.std(dist)/np.sqrt(len(dist)), np.mean(dist)+np.std(dist)/np.sqrt(len(dist))],'-',color=colors[index])
    plt.xticks(range(0,3),labels)
    plt.xlim(-1,3)
    plt.ylim(ylim)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(titlestr)
    if np.sum(df_flash_boots[0] > df_flash_boots[1])/len(df_flash_boots[0]) > (1-0.05):
        plt.plot(0.1,plt.gca().get_ylim()[1]*.9,'k*',markersize=10)
    if np.sum(df_cell_boots[0] > df_cell_boots[1])/len(df_cell_boots[0]) > (1-0.05):
        plt.plot(1.1,plt.gca().get_ylim()[1]*.9,'k*',markersize=10)
    
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



