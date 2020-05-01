import numpy as np
import psy_general_tools as pgt
import psy_tools as ps
import matplotlib.pyplot as plt
plt.ion()
import pandas as pd

def get_train_summary():
    train_summary = pd.read_csv('/home/alex.piet/codebase/behavior/model_output/_training_summary_table.csv')
    return train_summary

def plot_training_dropout(train_summary):
    '''
        train_summary is found in  _training_summary_table.csv
    '''
    donor_ids = train_summary.query('ophys & (stage > 0)').donor_id.unique()
    donor_vals = train_summary.query('ophys & (stage >0)').groupby('donor_id').task_dropout_index.mean().values

    plt.figure()
    plt.axhline(0,color='k',linestyle='--') 
    x = []
    y = []
    c = []
    for dex, donor_id in enumerate(donor_ids):
        mouse_table = train_summary.query('donor_id == @donor_id')
        vals = [mouse_table.query('(not ophys) & (stage == "3")').task_dropout_index.mean(),
        mouse_table.query('(not ophys) & (stage == "4")').task_dropout_index.mean(),
        mouse_table.query('(not ophys) & (stage == "5")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "0")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "1")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "3")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "4")').task_dropout_index.mean(),
        mouse_table.query('(ophys) & (stage == "6")').task_dropout_index.mean()]
        xvals = [-3,-2,-1,0,1,2,3,4]
        plt.plot(xvals, vals,'k-',alpha=.2)
        x = x + xvals
        y = y + vals
        c = c + list(np.ones(np.size(vals))*donor_vals[dex])

    scat = plt.gca().scatter(x, y, s=80,c =c, cmap='plasma',alpha=0.5)
    return x,y,c

