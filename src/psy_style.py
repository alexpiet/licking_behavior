import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def get_style():
    '''
        Returns a dictionary of fontsizes, alpha levels, etc

        Axline is a dashed line used to indicate a value like 0, or x/y symmetry
    '''
    style={
        'label_fontsize':16,
        'label_fontsize_dense':12,
        'axis_ticks_fontsize':12,
        'colorbar_label_fontsize':16,
        'colorbar_ticks_fontsize':12,
        'axline_color':'k',         
        'axline_alpha':0.5,
        'axline_linestyle':'--',
        'regression_color':'r',
        'regression_linestyle':'--',
        'annotation_color':'r',
        'annotation_linewidth':2,
        'annotation_alpha':1,
        'data_alpha':0.5,
        'data_color_all':'gray',#'tab:blue',
        'data_color_bias':'dimgray',#'tab:blue',
        'data_color_omissions':'tab:green',#mediumseagreen',#[0,158/256,115/256],#'tab:green',
        'data_color_omissions1':'magenta',#orchid',#[204/256,121/256,167/256],#'tab:red',
        'data_color_task0':'darkorange',#'tab:orange',
        'data_color_timing1D':'blue',#'tab:purple',
        'schematic_change': sns.color_palette()[0],
        'schematic_omission':sns.color_palette()[-1],
        'data_uncertainty_color':'k', # For use with fillbetween
        'data_uncertainty_alpha':0.15,# for single error bar use same data color
        'stats_color':'gray',
        'stats_alpha':1,
        'background_color':'k',
        'background_alpha':0.1
    }
    return style

def get_colors():
    tab10 = plt.get_cmap('tab10') 
    colors = {
        'Sst-IRES-Cre' :(158/255,218/255,229/255),
        'Vip-IRES-Cre' :(197/255,176/255,213/255),
        'Slc17a7-IRES2-Cre' :(255/255,152/255,150/255),
        'OPHYS_1_images_A':(148/255,29/255,39/255),
        'OPHYS_2_images_A':(222/255,73/255,70/255),
        'OPHYS_3_images_A':(239/255,169/255,150/255),
        'OPHYS_4_images_A':(43/255,80/255,144/255),
        'OPHYS_5_images_A':(100/255,152/255,193/255),
        'OPHYS_6_images_A':(195/255,216/255,232/255),
        'OPHYS_1_images_B':(148/255,29/255,39/255),
        'OPHYS_2_images_B':(222/255,73/255,70/255),
        'OPHYS_3_images_B':(239/255,169/255,150/255),
        'OPHYS_4_images_B':(43/255,80/255,144/255),
        'OPHYS_5_images_B':(100/255,152/255,193/255),
        'OPHYS_6_images_B':(195/255,216/255,232/255),
        'F1':(148/255,29/255,39/255),
        'F2':(222/255,73/255,70/255),
        'F3':(239/255,169/255,150/255),
        'N1':(43/255,80/255,144/255),
        'N2':(100/255,152/255,193/255),
        'N3':(195/255,216/255,232/255),
        '1':(148/255,29/255,39/255),
        '2':(222/255,73/255,70/255),
        '3':(239/255,169/255,150/255),
        '4':(43/255,80/255,144/255),
        '5':(100/255,152/255,193/255),
        '6':(195/255,216/255,232/255),
        '1.0':(148/255,29/255,39/255),
        '2.0':(222/255,73/255,70/255),
        '3.0':(239/255,169/255,150/255),
        '4.0':(43/255,80/255,144/255),
        '5.0':(100/255,152/255,193/255),
        '6.0':(195/255,216/255,232/255),
        1.0:(148/255,29/255,39/255),
        2.0:(222/255,73/255,70/255),
        3.0:(239/255,169/255,150/255),
        4.0:(43/255,80/255,144/255),
        5.0:(100/255,152/255,193/255),
        6.0:(195/255,216/255,232/255),
        'Familiar':(0.66,0.06,0.086),
        'Novel 1':(0.044,0.33,0.62),
        'Novel >1':(0.34,.17,0.57),
        'engaged':'lightgray',
        'disengaged':'dimgray',
        'visual engaged':'darkorange',
        'timing engaged':'blue',
        'visual disengaged':'burlywood',
        'timing disengaged':'lightblue',
        'reward_rate':'m',
        'lick_bout_rate':'g',
        'visual':'darkorange',
        'timing':'blue',
        'visual sessions':'darkorange',
        'timing sessions':'blue',
        'no strategy':'green',
        'mixed':'green',
        'bias':'dimgray',
        'omissions':'tab:green',
        'omissions1':'magenta',
        'task0':'darkorange',
        'timing1D':'blue',
    }
    return colors

def get_project_colors(keys=None):
    '''
        Returns a dictionary of colors
        keys is a list. For each element of keys, if its not defined
            it is assigned a random tab10 color 
    '''
    tab10 = plt.get_cmap('tab10') 
    colors = get_colors()
    
    if keys is not None:
        for index, key in enumerate(keys):
            if key not in colors:
                colors[key] = tab10(np.mod(index,10))
    return colors


