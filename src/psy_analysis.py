import numpy as np
import matplotlib.pyplot as plt

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





