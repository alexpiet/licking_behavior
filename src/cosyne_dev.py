import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def get_pivot():
    manifest = pd.read_csv('/allen/programs/braintv/workgroups/nc-ophys/visual_behavior/behavior_model_output/_summary_table.csv')

    x = manifest[['specimen_id','stage','task_dropout_index']]
    x_pivot = pd.pivot_table(x,values='task_dropout_index',index='specimen_id',columns=['stage'])
    x_pivot['mean_index'] = [np.nanmean(x) for x in zip(x_pivot[1],x_pivot[3],x_pivot[4],x_pivot[6])]

    x_pivot['mean_1'] = x_pivot[1] - x_pivot['mean_index']
    x_pivot['mean_3'] = x_pivot[3] - x_pivot['mean_index']
    x_pivot['mean_4'] = x_pivot[4] - x_pivot['mean_index']
    x_pivot['mean_6'] = x_pivot[6] - x_pivot['mean_index']
    return x_pivot

def plot(x_pivot, w=.45):
    plt.figure(figsize=(5,5))
    stages = [1,3,4,6]
    counts = [1,2,3,4]

    for val in zip(counts,stages):
        m = x_pivot['mean_'+str(val[1])].mean()
        s = x_pivot['mean_'+str(val[1])].std()/np.sqrt(len(x_pivot))
        plt.plot([val[0]-w,val[0]+w],[m,m],linewidth=4)
        plt.plot([val[0],val[0]],[m+s,m-s],linewidth=1)

    plt.ylabel('$\Delta$ Strategy Index',fontsize=24)
    plt.xlabel('Session #',fontsize=24)
    plt.yticks(fontsize=16)
    plt.xticks(counts,['F1','F3','N1','N3'],fontsize=24)
    plt.gca().axhline(0,color='k',linestyle='--',alpha=.25)
    plt.tight_layout()
    


