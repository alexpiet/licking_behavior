from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from licking_behavior.src import licking_model as mo
from tqdm import tqdm
import sys; sys.path.append('/home/nick.ponvert/src/nick-allen')
import colorpalette

fn = '/home/nick.ponvert/model_fits.h5'

model_fits = pd.read_hdf(fn, key='df')
model_fits = model_fits.query('hit_trial_count > 0').copy()

# TODO: This doesn't work due to storing list in cell problems
for ind_row, row in model_fits.iloc[11:12].iterrows():
    print('row {}'.format(ind_row))
    model = mo.unpickle_model(row['model_file'])
    lick_inds = np.flatnonzero(model.filters['post_lick'].data)
    overall_predictability = model.calculate_latent()[lick_inds]

    filter_linear = {}
    filter_nonlinear = {}
    for filter_name, filter_obj in model.filters.items():
        this_filter_linear = filter_obj.linear_output()[lick_inds]
        this_filter_nonlinear = np.exp(filter_obj.linear_output())[lick_inds]
        filter_linear.update({filter_name:this_filter_linear})
        filter_nonlinear.update({"{}_nonlinear".format(filter_name):this_filter_nonlinear})

    d = {'lick_ind':lick_inds,
         'overall_predict':overall_predictability}
    d.update(filter_linear)
    d.update(filter_nonlinear)
    df = pd.DataFrame(d)

from sklearn.manifold import TSNE

filter_predict = df[['post_lick', 'reward', 'flash', 'change_flash']].values
filter_plot = df[['post_lick_nonlinear', 'reward_nonlinear', 'flash_nonlinear', 'change_flash_nonlinear']].values

tsne = TSNE(verbose=1)
X_new = tsne.fit_transform(filter_plot)

plt.subplot(2, 2, 1)
plt.scatter(X_new[:,0], X_new[:,1], c=filter_plot[:,0])
cbar = plt.colorbar()
plt.clim([0, 20])
cbar.set_label('Post-lick filter gain')

plt.subplot(2, 2, 2)
plt.scatter(X_new[:,0], X_new[:,1], c=filter_plot[:,1])
cbar = plt.colorbar()
cbar.set_label('Reward filter gain')

plt.subplot(2, 2, 3)
plt.scatter(X_new[:,0], X_new[:,1], c=filter_plot[:,2])
cbar = plt.colorbar()
cbar.set_label('Flash filter gain')

plt.subplot(2, 2, 4)
plt.scatter(X_new[:,0], X_new[:,1], c=filter_plot[:,3])
cbar =  plt.colorbar()
cbar.set_label('Change flash filter gain')
#  plt.plot(X_new[:,0], X_new[:,1], '.')
plt.show()






#  from sklearn.cluster import DBSCAN

#  from sklearn.cluster import AgglomerativeClustering
#  clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(filter_predict)

#  from sklearn.cluster import KMeans
#  sse = np.empty(10)
#  for ind_k, k in enumerate(range(1, 11)):
#      kmeans = KMeans(n_clusters=k, max_iter=1000).fit(filter_predict)
#      sse[ind_k] = kmeans.inertia_
#  
#  plt.figure()
#  plt.plot(sse)
#  ax=plt.gca()
#  ax.set_xticks(range(0, 10))
#  ax.set_xticklabels(range(1, 11))
#  plt.xlabel("k")
#  plt.ylabel("SSE")

#  dbscan = DBSCAN()
#  clustering = DBSCAN(eps=0.1, min_samples=2).fit(filter_predict)

#  kmeans = KMeans(n_clusters=4, max_iter=1000).fit(filter_predict)

labels = np.argmax(filter_predict, axis=1)
labels = np.argmax(filter_plot, axis=1)
colors = colorpalette.get_colors(4)

plt.figure()
for ind_label, label in enumerate(['post_lick', 'reward', 'flash', 'change_flash']):
    points_this_label = labels==ind_label
    plt.scatter(X_new[points_this_label,0], X_new[points_this_label,1], c=colors[ind_label], label=label)
plt.legend()
plt.xlabel('Dim 0')
plt.ylabel('Dim 1')
plt.show()

