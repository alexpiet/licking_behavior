import pickle
from alex_utils import load
import matplotlib.pyplot as plt
import fit_tools
plt.ion()
import numpy as np

filepath = '/Users/alex.piet/glm_fits/'

experiment_ids = [837729902, 838849930,836910438,840705705,840157581,841601446,840702910,841948542,841951447,842513687,842973730,843519218,846490568,847125577,848697604] 
dt = 0.01

plt.close('all')
ids = experiment_ids[-1]
res = load(filepath+'fitglm_'+str(ids))
dt = 0.01
data = fit_tools.get_sdk_data(ids, load_dir='/allen/aibs/technology/nicholasc/behavior_ophys')
licks, licksdt, start_time, stop_time, time_vec, running_speed, rewardsdt, flashesdt, change_flashesdt = fit_tools.extract_sdk_data(data,dt)

flashes = flashesdt/100
rewards = rewardsdt/100
change_flashes = change_flashesdt/100

 
fit_tools.compare_model(res.latent, time_vec, licks, stop_time, rewards=rewards, flashes=flashes, change_flashes=change_flashes)




