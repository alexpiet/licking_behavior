import os
import numpy as np
import pandas as pd
import collections
import seaborn as sns

import matplotlib.pyplot as plt
from visual_behavior.ophys.dataset.visual_behavior_ophys_dataset import VisualBehaviorOphysDataset
from visual_behavior.ophys.response_analysis.response_analysis import ResponseAnalysis


cache_dir = r'\\allen\programs\braintv\workgroups\nc-ophys\visual_behavior\visual_behavior_production_analysis'
def gen_interlick_df(experiment_list, rew_trig_window= 5):
    """[takes a list of experients, calculates & generates a dataframe with inter lick interval times within a time window of a reward]
    
    Arguments:
       experiment_list {[list]} -- [list of experiments to compile a dataframe for]
    
    Keyword Arguments:
        rew_trig_window {int} -- [window of time in seconds, triggered by reward, to 
            generate inter-lick interval times for] (default: {5})
    
    Returns:
        [dataframe] -- [pandas dataframe with following columns:
                        ["mouse", experiment_id", "reward_time","rew_num", "lick_time", "inter_lick_interval"]
    """
    
    ########### load and compile the data to plot ##############
    
    #create an empty dataframe to fill
    reward_interlick_df= pd.DataFrame(columns=["mouse_id","experiment_id", "reward_time","rew_num", "lick_time", "inter_lick_interval"])
    
    ###go through by experiment and extract the pertinent info
    
    experiment_counter = 1
    experiment_set = set(experiment_list)
    for experiment in experiment_set:
        experiment_id = experiment
        
        #load data with vba package
        
        dataset= VisualBehaviorOphysDataset(experiment_id, cache_dir=cache_dir)
        
        mouse= dataset.metadata.donor_id[0]
        licks = dataset.licks
        licks.rename(index=str, columns={"time": "lick_time"}, inplace = True)
        reward_times = dataset.rewards.time.values
        
        #printing statements
        print("on experiment " + str(experiment_counter) +" of " +str(len(experiment_set)))
        experiment_counter = experiment_counter + 1

        ## go through by reward times and & extract & calculate 
        
        rew_counter = 1
        for reward_time in reward_times:
            current_rew = reward_time
            #create shell df with session id current reward time & number
            rew_lick_df = licks.copy()
            rew_lick_df["reward_time"] = current_rew
            rew_lick_df["experiment_id"] = experiment_id
            rew_lick_df["rew_num"] = rew_counter
            rew_lick_df["mouse_id"] = mouse
            
            print("on reward " + str(rew_counter) +" of " +str(len(reward_times)))
            rew_counter = rew_counter + 1
            
            #create the time window 
            win_start = current_rew
            win_end = win_start + 5
            
            #get licks within a 5 second window of a reward
            rew_lick_df["within_rew_win"] = rew_lick_df["lick_time"].between(win_start, win_end, inclusive = True)
            rew_lick_df = rew_lick_df.loc[rew_lick_df["within_rew_win"]==True]

            #get inter lick times
            rew_lick_df["inter_lick_interval"]= rew_lick_df.lick_time.diff()
            
            #drop un-needed columns
            rew_lick_df = rew_lick_df.drop(["within_rew_win"],axis=1)
            rew_lick_df = rew_lick_df.drop(["frame"], axis=1)
            
            #append main dataframe with reward level df
            reward_interlick_df = reward_interlick_df.append(rew_lick_df)
            
            #drop nan rows (first lick after reward)
            reward_interlick_df = reward_interlick_df.dropna()
    
    return reward_interlick_df