import psy_tools as ps
import psy_timing_tools as pt
import psy_metrics_tools as pm
import matplotlib.pyplot as plt
import psy_cluster as pc
from alex_utils import *
from importlib import reload
plt.ion()
import pandas as pd
from sklearn import linear_model
from sklearn import model_selection

def analyze_manifest(manifest):
    plt.figure()
    manifest.groupby(driver_line)
    # Average by cre-line
    # Average over all?
    # make matrix of coefficients?


def run_manifest(manifest):
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

def run_session(id,plot_this=True,verbose=False):
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

def get_cell_flash_df(cell_id,session,fit,all_stim=False):
    cell_flash_df = get_cell_df(cell_id,session)
    cell_flash_df = add_metrics(cell_flash_df,session)
    cell_flash_df = remove_licking_bout(cell_flash_df,fit)
    cell_flash_df = add_model_weights(cell_flash_df,fit)
    if not all_stim:
        cell_flash_df = cell_flash_df[cell_flash_df['pref_stim']] 
    return cell_flash_df

def run_cell_ridge(cell_id,session,fit,verbose=False,use_metrics=True,use_model=True,use_task=True,plot_cell=False,use_full=False):
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

def run_cell_ols(cell_id,session,fit,verbose=True,use_task_state=True):
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

def get_cell_df(cell_id, session):
    return session.flash_response_df[session.flash_response_df['cell_specimen_id'] == cell_id]

def add_metrics(cell_flash_df,session):
    cell_flash_df = cell_flash_df.assign(bout_rate = session.stimulus_presentations['bout_rate'].values)
    cell_flash_df = cell_flash_df.assign(reward_rate = session.stimulus_presentations['reward_rate'].values)
    return cell_flash_df

def remove_licking_bout(cell_flash_df,fit):
    cell_flash_df = cell_flash_df.assign(in_bout = fit['psydata']['full_df']['in_bout'].values)
    cell_flash_df = cell_flash_df[cell_flash_df['in_bout'] != 1]
    return cell_flash_df

def add_model_weights(cell_flash_df,fit):
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

def build_design_matrix(cell_flash_df,regressor_list,not_change=True):
    if not_change:
        temp = cell_flash_df[~cell_flash_df['change']]
    else:
        temp = cell_flash_df
    design_matrix = temp[regressor_list].to_numpy()
    y = temp.mean_response.values
    return design_matrix,y



def plot_summary(id):
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

def get_session_change_modulation(id):
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

def get_cell_change_modulation(cell,session,fit):
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

def plot_cell_change_modulation(cm,full,task,bias):
    fig,ax = plt.subplots(nrows=3,ncols=1)
    ax[0].plot(full, cm,'ko')
    ax[1].plot(task, cm,'ko')
    ax[2].plot(bias, cm,'ko')

def is_cell_significant(cell_flash_df):
    return (np.sum(cell_flash_df['p_value'] < 0.05)/len(cell_flash_df)) > 0.25


def streamline_get_session_change_modulation(id):
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

def streamline_get_cell_change_modulation(cell,session):
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

def flash_hit(flash_id,session):
    return len(session.stimulus_presentations.loc[flash_id]['rewards']) > 0
