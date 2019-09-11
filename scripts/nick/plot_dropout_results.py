import pandas as pd
import os
import glob
from licking_behavior.src import licking_model as mo

model_fits = pd.read_hdf('/home/nick.ponvert/nco_home/data/20190714_model_fits.h5', key='df')

for ind_row, row in model_fits.iloc[1:2].iterrows():
    model = mo.unpickle_model(row['model_file'])

