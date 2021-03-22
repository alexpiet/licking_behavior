import matplotlib.pyplot as plt
import numpy as np

def get_project_colors(keys=None):
    colors = {
        'Sst-IRES-Cre' : (158/255,218/255,229/255),
        'Vip-IRES-Cre' : (197/255,176/255,213/255),
        'Slc17a7-IRES2-Cre' : (255/255,152/255,150/255),
        'Sst' : (158/255,218/255,229/255),
        'Vip' : (197/255,176/255,213/255),
        'Slc' : (255/255,152/255,150/255),
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
        'N3':(195/255,216/255,232/255)

    }

    if keys is not None:
        tab10 = plt.get_cmap('tab10') 
        for index, key in enumerate(keys):
            if key not in colors:
                colors[key] = tab10(np.mod(index,10))
    return colors


