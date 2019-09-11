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


fn = '/home/nick.ponvert/model_fits.h5'

model_fits = pd.read_hdf(fn, key='df')
model_fits = model_fits.query('hit_trial_count > 0').copy()

all_lick_mean = np.empty(len(model_fits))
all_no_lick_mean = np.empty(len(model_fits))
for ind_iter, (ind_row, row) in enumerate(model_fits.iterrows()):
    print('row {}'.format(ind_row))
    model = mo.unpickle_model(row['model_file'])
    lick = model.filters['post_lick'].data.astype(bool)
    lick_predict_all = model.calculate_latent()[lick]
    no_lick_predict_all = model.calculate_latent()[np.logical_not(lick)]
    lick_predict_mean = model.calculate_latent()[lick].mean()
    no_lick_predict_mean = model.calculate_latent()[np.logical_not(lick)].mean()
    all_lick_mean[ind_iter] = lick_predict_mean
    all_no_lick_mean[ind_iter] = no_lick_predict_mean

plt.clf()
plt.plot(all_no_lick_mean, all_lick_mean, 'k.', ms=12)
lims = [-0.001, 0.121]
plt.xlim(lims)
plt.ylim(lims)
plt.plot(lims, lims, '--', color='0.5')
plt.gca().set_aspect(aspect='equal')
plt.xlabel('Mean P(lick | no lick)')
plt.ylabel('Mean P(lick | lick)')

plt.clf()
plt.hist(all_lick_mean / all_no_lick_mean, histtype='step', color='k', lw=2)
extraplots.boxoff(plt.gca())
plt.xlabel('P(lick | lick) / P(lick | no lick)')
plt.ylabel('Count')
plt.xlim([0, 45]
plt.tight_layout()
plt.show()

