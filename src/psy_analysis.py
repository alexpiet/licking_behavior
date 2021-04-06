
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def RT_by_engagement(ophys,version=None,bins=45,title=''):
    plt.figure()
    RT_engaged = []
    for index, row in ophys.iterrows():
        RT_engaged.append(row['RT'][row['engaged'].astype(bool)])
    RT_disengaged = []
    for index, row in ophys.iterrows():
        RT_disengaged.append(row['RT'][~row['engaged'].astype(bool)])
       
    RT_engaged = np.hstack(RT_engaged)
    RT_disengaged = np.hstack(RT_disengaged)
    plt.hist(RT_engaged, color='k',alpha=.5,label='Engaged',bins=bins)
    plt.hist(RT_disengaged, color='r',alpha=.5,label='Disengaged',bins=bins)
    plt.ylabel('count',fontsize=16)
    plt.xlabel('RT (s)',fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlim(0,.75)
    plt.legend()
    plt.title(title)
    plt.tight_layout()

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





