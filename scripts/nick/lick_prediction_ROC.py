from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from licking_behavior.src import licking_model as mo
from tqdm import tqdm
import sys; sys.path.append('/home/nick.ponvert/src/nick-allen')
import colorpalette
import extraplots
import seaborn as sns
sns.set_context('talk')

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fn = '/home/nick.ponvert/model_fits.h5'
model_fits = pd.read_hdf(fn, key='df')
model_fits = model_fits.query('hit_trial_count > 0').copy()

auc_scores = np.empty(len(model_fits))
for ind_iter, (ind_row, row) in enumerate(model_fits.iterrows()):
    print('row {}'.format(ind_row))
    model = mo.unpickle_model(row['model_file'])
    predicted = model.calculate_latent()
    actual = model.filters['post_lick'].data
    if actual.sum() > 0:
        auroc = roc_auc_score(actual, predicted)
    else:
        auroc = 0
    auc_scores[ind_iter] = auroc

far, hr, thresh = roc_curve(actual, predicted)
plt.clf()
plt.plot(far, hr, 'k-')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.gca().set_aspect('equal')
plt.xlabel("False alarm rate")
plt.ylabel("Hit rate")
plt.show()


plt.clf()
plt.hist(auc_scores, color='k', bins = np.arange(0.5, 1, 0.05), histtype='step', lw=2)
extraplots.boxoff(plt.gca())
plt.xlim([0.5, 1])
plt.ylabel('Count')
plt.xlabel('AUROC')
plt.tight_layout()
