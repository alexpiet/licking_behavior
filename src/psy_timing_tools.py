import psy_general_tools as pgt
import matplotlib.pyplot as plt
import psy_metrics_tools as pm
import numpy as np
import seaborn
import pandas as pd
import matplotlib.patches as patches
from tqdm import tqdm

# TODO, Issue #234
def plot_all_session_interlick_distributions(summary_df,version, savefig=False):
    '''
        Generate all single session plots of the licking distributions
        # TODO, have it make all relevant plots here while the session object
        is loaded
    '''
    num_crash = 0
    for bsid in tqdm(summary_df['behavior_session_id']):
        try:
            session = pgt.get_data(bsid)
            plot_session_interlick_interval_distribution(
                session,version=version,savefig=savefig)
        except:
            num_crash +=1
        plt.close('all')
    print(str(num_crash)+' sessions crashed')

# TODO, Issue #234
def plot_session_interlick_interval_distribution(session,nbins=50,savefig=False):
    '''
        Plots the distribution of all licks, and hit and miss licks
        # TODO
        # save directory
        # style guidelines
        # super-impose hit and miss
        # QC on how I compute things
        # separate computation and plotting
        # can this operate off the session_df?
            Just add a list of lick times to the summary_df?
    '''
    #pm.annotate_licks(session)
    licks = session.licks.timestamps.values
    diffs = np.diff(licks) 
    fig, ax = plt.subplots(1,2,figsize=(12,5))
    h=ax[0].hist(diffs[diffs<10],nbins,label='All')
    ax[0].axvline(0.75,linestyle='--',color='k')
    ax[0].set_ylabel('count')
    ax[0].set_xlabel('InterLick (s)')
    ax[0].set_ylim([0,100])
    #ax[0].set_title(str(session.metadata['mouse_id'])+" "+session.metadata['stage'])
    m = get_mean_lick_distribution(session)
    ax[0].axvline(m,linestyle='--',color='r')
    d = session.licks['pre_ili'][session.licks.rewarded]
    h2= ax[0].hist(d[(d>.7)&(d<10)],bins=h[1],label='Hits')
    ax[0].legend()
    h1 = np.histogram(diffs[(diffs>.7)&(diffs<10)],bins=h[1])
    centers = np.diff(h[1]) + h[1][0:-1]
    ax[1].plot(centers[centers > .7], (h1[0]-h2[0])[centers > .7]  ,'k-')
    ax[1].set_ylabel('Miss Licks')
    ax[1].set_xlabel('InterLick (s)')
    if savefig:
        bsid = session.metadata['ophys_experiment_id']
        plt.savefig(directory+str(bsid)+"_ILI.svg")

# TODO, Issue #234
def plot_all_mouse_durations(all_durs,directory=None):
    plt.figure()
    for dur in all_durs:
        #if len(dur) == 4:
        plt.plot(dur,'o-')
    plt.ylabel('Avg IBI (s)')
    plt.xlabel('Ophys Session #')
    plt.ylim(bottom=0)
    if type(directory) is not type(None):
        plt.savefig(directory+"avg_ILI_by_session.svg")

# TODO, Issue #234
def get_all_mouse_durations(mice_ids):
    all_durs=[]
    for mouse in tqdm(mice_ids):
        print(str(mouse))
        try:
            durs = get_mouse_durations(mouse)
            all_durs.append(durs)
        except Exception as e:
            print("  crash "+str(e))
    return all_durs

# TODO, Issue #234
def get_mouse_durations(mouse_id):
    sessions,IDS,active = pgt.load_mouse(mouse_id)
    durs = []
    for sess in np.array(sessions)[active]:
        durs.append(get_mean_lick_distribution(sess))
    return durs

# TODO, Issue #234
def get_mean_lick_distribution(session,threshold=20):
    pm.annotate_licks(session)
    diffs = session.licks[session.licks['bout_start']]['pre_ili']
    return np.mean(diffs[diffs < threshold])

# TODO, Issue #233
def get_bout_table(session):
    bout = session.licks.groupby('bout_number').apply(len).to_frame()
    bout = bout.rename(columns={0:"length"})
    bout['bout_rewarded'] = session.licks.groupby('bout_number').any('rewarded')['bout_rewarded']
    bout['bout_duration'] = session.licks.groupby('bout_number').last()['timestamps'] - session.licks.groupby('bout_number').first()['timestamps']
    bout['post_ili'] = session.licks.groupby('bout_number').last()['post_ili']
    bout['post_ili_from_start'] = session.licks.groupby('bout_number').last()['post_ili'] + bout['bout_duration']
    bout['pre_ili'] = session.licks.groupby('bout_number').first()['pre_ili']
    bout['pre_ili_from_start'] = session.licks.groupby('bout_number').first()['pre_ili'] + bout['bout_duration'].shift(1)
    return bout

# TODO, Issue #233
def plot_bout_ili(bout,from_start=False,directory=None):
    plt.figure()
    if from_start:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'], bins = 25,color='k',label='post-miss')
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'],bins=h[1], color='r',label='post-reward')
    else:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'], bins = 25,color='k',label='post-miss')
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'],bins=h[1], color='r',label='post-reward')
    plt.legend()
    plt.ylabel('count',fontsize=12)
    if from_start:
        plt.xlabel('InterLick (s), time from bout start', fontsize=12)
    else:
        plt.xlabel('InterLick (s), time from bout end', fontsize=12)
    ylims = plt.ylim()
    mean_all, mean_reward = get_bout_ili(bout, from_start=from_start)
    plt.plot(mean_all, ylims[1], 'kv')    
    plt.plot(mean_reward, ylims[1], 'rv') 
    if type(directory) is not type(None):
        if from_start:
            plt.savefig(directory+"bout_ili_distribution_from_start.svg")
        else:
            plt.savefig(directory+"bout_ili_distribution_from_end.svg")
# TODO, Issue #233
def plot_bout_ili_current(bout,from_start=False,directory=None):
    plt.figure()
    if from_start:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili_from_start'], bins = 25,color='b',label='post-miss miss',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili_from_start'],bins=h[1], color='m',label='post-reward miss',alpha=.25)
        plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili_from_start'],bins=h[1], color='k',label='post-miss hit',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili_from_start'],bins=h[1], color='r',label='post-reward hit',alpha=.25)
    else:
        h = plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili'], bins = 25,color='b',label='post-miss miss',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili'],bins=h[1], color='m',label='post-reward miss',alpha=.25)
        plt.hist(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili'],bins=h[1], color='k',label='post-miss hit',alpha=.25)
        plt.hist(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)&(bout['bout_rewarded'].shift(-1,fill_value=True))]['post_ili'],bins=h[1], color='r',label='post-reward hit',alpha=.25)
    plt.legend()
    plt.ylabel('count',fontsize=12)
    if from_start:
        plt.xlabel('InterLick (s), time from bout start', fontsize=12)
    else:
        plt.xlabel('InterLick (s), time from bout end', fontsize=12)
    ylims = plt.ylim()
    mean_miss, mean_reward = get_bout_ili_current(bout, from_start=from_start,current_hit=True)
    plt.plot(mean_miss, ylims[1], 'kv')    
    plt.plot(mean_reward, ylims[1], 'rv') 
    mean_miss, mean_reward = get_bout_ili_current(bout, from_start=from_start,current_hit=False)
    plt.plot(mean_miss, ylims[1], 'b^')    
    plt.plot(mean_reward, ylims[1], 'm^') 
    if type(directory) is not type(None):
        if from_start:
            plt.savefig(directory+"bout_ili_distribution_from_start_current_hitmiss.svg")
        else:
            plt.savefig(directory+"bout_ili_distribution_from_end_current_hitmiss.svg")

 # TODO, Issue #233
def get_bout_ili(bout, from_start=False):
    if from_start:
        mean_miss = np.nanmean(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'])
        mean_reward = np.nanmean(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili_from_start'])
    else:
        mean_miss = np.nanmean(bout[(~bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'])
        mean_reward = np.nanmean(bout[(bout['bout_rewarded']) & (bout['post_ili'] < 10)]['post_ili'])
    return mean_miss, mean_reward

# TODO, Issue #233
def get_bout_ili_current(bout,from_start=False, current_hit=True):
    if from_start:
        start_str = 'post_ili_from_start'
    else:
        start_str = 'post_ili'
    if current_hit:
        mean_miss = np.nanmean( bout[(~bout['bout_rewarded'])&(bout['post_ili']<10)&(bout['bout_rewarded'].shift(-1))][start_str])
        mean_reward = np.nanmean( bout[(bout['bout_rewarded'])&(bout['post_ili']<10)&(bout['bout_rewarded'].shift(-1))][start_str])
    else:
        mean_miss = np.nanmean( bout[(~bout['bout_rewarded'])&(bout['post_ili']<10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))][start_str])
        mean_reward = np.nanmean( bout[(bout['bout_rewarded'])&(bout['post_ili']<10)&(~bout['bout_rewarded'].shift(-1,fill_value=True))][start_str])  
    return mean_miss, mean_reward

# TODO, Issue #233
def plot_bout_durations(bout,directory=None,alpha=0.5):
    plt.figure()
    aweights = np.ones_like(bout.query('not bout_rewarded')['length'].values)/float(len(bout))
    rweights = np.ones_like(bout.query('bout_rewarded')['length'].values)/float(len(bout))
    h = plt.hist(bout.query('not bout_rewarded')['length'],bins=np.max(bout['length']),color='k',label='Not-Rewarded',alpha=alpha,weights=aweights)
    plt.hist(bout.query('bout_rewarded')['length'],bins=h[1],color='r',label='Rewarded',alpha=alpha,weights=rweights)
    plt.xlabel('# Licks in bout',fontsize=12)
    plt.ylabel('Prob',fontsize=12)
    plt.legend()
    plt.gca().set_xticks(np.arange(0,np.max(bout['length']),5))
    plt.xlim(0,35)
    plt.tight_layout()
    if type(directory) is not type(None):
        plt.savefig(directory+"licks_in_bouts.svg")

    plt.figure()
    h = plt.hist(bout.query('not bout_rewarded')['length'],bins=np.max(bout['length']),color='k',label='Not-Rewarded',alpha=alpha,density=True)
    plt.hist(bout.query('bout_rewarded')['length'],bins=h[1],color='r',label='Rewarded',alpha=alpha,density=True)
    plt.xlabel('# Licks in bout',fontsize=24)
    plt.ylabel('Prob | Reward Type.',fontsize=24)
    plt.legend()
    plt.gca().set_xticks(np.arange(0,np.max(bout['length']),5))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlim(0,35)
    plt.tight_layout()
    if type(directory) is not type(None):
        plt.savefig(directory+"licks_in_bouts_normalized.svg")


    plt.figure()
    h = plt.hist(bout.query('not bout_rewarded')['bout_duration'],bins=np.max(bout['length']),color='k',label='Not-Rewarded',alpha=alpha)
    plt.hist(bout.query('bout_rewarded')['bout_duration'],bins=h[1],color='r',label='Rewarded',alpha=alpha)
    plt.xlabel('bout duration (s)',fontsize=12)
    plt.ylabel('Count',fontsize=12)
    plt.legend()
    plt.gca().set_xticks(np.arange(0,np.max(bout['bout_duration']),0.5))
    plt.xlim(0,5)
    if type(directory) is not type(None):
        plt.savefig(directory+"bout_duration.svg")

# TODO, Issue #233
def get_all_bout_table(IDS):
    session = pgt.get_data(IDS[0])
    pm.annotate_licks(session)
    all_bout = get_bout_table(session)
    for id in IDS[1:]:
        print(id)
        try:
            session = pgt.get_data(id)
            pm.annotate_licks(session)
            bout = get_bout_table(session)
            all_bout = pd.concat([all_bout, bout])    
        except Exception as e:
            print(" crash "+str(e))
    return all_bout    

# TODO, Issue #233
def get_all_bout_statistics(IDS):
    durs = []
    for id in IDS:
        print(id)
        try:
            session = pgt.get_data(id)
            pm.annotate_licks(session)
            bout = get_bout_table(session) 
            my_durs = np.concatenate([get_bout_ili(bout, from_start=False), get_bout_ili(bout,from_start=True), get_bout_ili_current(bout, from_start=False,current_hit=False),get_bout_ili_current(bout, from_start=True,current_hit=False),get_bout_ili_current(bout, from_start=False,current_hit=True),get_bout_ili_current(bout, from_start=True,current_hit=True)])
            durs.append(my_durs)
        except Exception as e:
            print(" crash "+str(e))
    return durs    

# TODO, Issue #233
def plot_all_bout_statistics(durs,all_bout=None,directory=None):
    # the mean post miss is 4 flashes + 2.3 flasshes
    # 2.3 =mean(geometric(.3)) = 1/.3
    # each flash is 0.75 seconds, and you want to respond anywhere in that range so:
    optimal_post_miss = [5.3*.75+.3, 6.3*.75+.3+.75]  
    # Post hit,  is 3.75 to 4.5 seconds of grace period, then the optimal post_miss
    optimal_post_hit = [3.75 + 5.3*.75, 4.5 + 6.3*.75]
    durs = np.vstack(durs)
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(durs[:,2], durs[:,3], 'ko',alpha=.2)
    ax[0].plot([0,10],[0,10],'k--',alpha=.2)
    ax[0].set_ylabel('mean IBI (s), post-hit \n from start of bout',fontsize=12)
    ax[0].set_xlabel('mean IBI (s), post-miss \n from start of bout',fontsize=12)
    ax[1].plot(durs[:,0], durs[:,1], 'ko',alpha=.2)
    ax[1].plot([0,10],[0,10],'k--',alpha=.2)
    ax[1].set_ylabel('mean IBI (s), post-hit \n from end of bout',fontsize=12)
    ax[1].set_xlabel('mean IBI (s), post-miss \n from end of bout',fontsize=12)
    plt.tight_layout()
    if type(all_bout) is not type(None):
        from_start_durs = get_bout_ili(all_bout, from_start=True)
        ax[0].plot(from_start_durs[0], from_start_durs[1],'ko')
        from_end_durs = get_bout_ili(all_bout, from_start=False)
        ax[1].plot(from_end_durs[0], from_end_durs[1],'ko')
    x = optimal_post_miss[0]
    xsize = np.diff(optimal_post_miss)
    y = optimal_post_hit[0]
    ysize = np.diff(optimal_post_hit)
    rect = patches.Rectangle((x,y),xsize,ysize,linewidth=1, edgecolor='m', facecolor='m',alpha=.25)
    ax[0].add_patch(rect)
    if type(directory) is not type(None):
        plt.savefig(directory+"bout_ili_scatter.svg")

# TODO, Issue #233
def plot_all_bout_statistics_current(durs,all_bout=None,directory=None):
    optimal_post_miss = [5.3*.75+.3, 6.3*.75+.3+.75]  
    optimal_post_hit = [3.75 + 5.3*.75, 4.5 + 6.3*.75]
    durs = np.vstack(durs)
    fig,ax = plt.subplots(1,2,figsize=(10,5))
    ax[0].plot(durs[:,6], durs[:,7], 'bo',alpha=.2)
    ax[0].plot(durs[:,10], durs[:,11], 'ro',alpha=.2)
    ax[0].plot([0,10],[0,10],'k--',alpha=.2)
    ax[0].set_ylabel('mean IBI (s) post-hit \n from start of bout',fontsize=12)
    ax[0].set_xlabel('mean IBI (s) post-miss \n from start of bout',fontsize=12)
    ax[1].plot(durs[:,4], durs[:,5], 'bo',alpha=.2)
    ax[1].plot(durs[:,8], durs[:,9], 'ro',alpha=.2)
    ax[1].plot([0,10],[0,10],'k--',alpha=.2)
    ax[1].set_ylabel('mean IBI (s) post-hit \n from end of bout',fontsize=12)
    ax[1].set_xlabel('mean IBI (s) post-miss \n from end of bout',fontsize=12)
    plt.tight_layout()
    if type(all_bout) is not type(None):
        from_start_durs = get_bout_ili_current(all_bout, from_start=True,current_hit=True)
        ax[0].plot(from_start_durs[0], from_start_durs[1],'ro',label='Current Hit')
        from_end_durs = get_bout_ili_current(all_bout, from_start=False,current_hit=True)
        ax[1].plot(from_end_durs[0], from_end_durs[1],'ro')

        from_start_durs_miss = get_bout_ili_current(all_bout, from_start=True,current_hit=False)
        ax[0].plot(from_start_durs_miss[0], from_start_durs_miss[1],'bo',label='Current Miss')
        from_end_durs_miss = get_bout_ili_current(all_bout, from_start=False,current_hit=False)
        ax[1].plot(from_end_durs_miss[0], from_end_durs_miss[1],'bo')
    x = optimal_post_miss[0]
    xsize = np.diff(optimal_post_miss)
    y = optimal_post_hit[0]
    ysize = np.diff(optimal_post_hit)
    rect = patches.Rectangle((x,y),xsize,ysize,linewidth=1, edgecolor='m', facecolor='m',alpha=.25)
    ax[0].add_patch(rect)
    ax[0].legend()    
    if type(directory) is not type(None):
        plt.savefig(directory+"bout_ili_scatter_by_current_hitmiss.svg")
