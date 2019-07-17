from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from licking_behavior.src import licking_model as mo
from tqdm import tqdm

fn = '/home/nick.ponvert/model_fits.h5'

model_fits = pd.read_hdf(fn, key='df')

mFn = model_fits.iloc[0]['model_file']

# TODO: This doesn't work due to storing list in cell problems
for ind_row, row in model_fits.iterrows():
    print('row {}'.format(ind_row))
    model = mo.unpickle_model(mFn)
    for filter_name, filter_obj in model.filters.items():
        nonlinear_filt = np.exp(filter_obj.build_filter())
    # Save it to the dataframe
    model_fits.at[ind_row, filter_name] = [nonlinear_filt.astype(list)]
