import numpy as np
from licking_behavior.src import licking_model as mo
from licking_behavior.src import fit_tools
from licking_behavior.src import filters
import importlib; importlib.reload(filters)
from matplotlib import pyplot as plt

def training_set_masks(arr, n_folds):
    '''
    arr should be one-dim
    output is an array of shape (n_folds, len(arr)) with 1 for training set and 0 for test
    '''
    # Use each split number as the held-out portion
    split_number = np.random.randint(0, n_folds, size=len(arr))
    training_set_masks = np.empty((n_folds, len(arr)), dtype=bool)
    for ind_split in range(n_folds):
        training_set_masks[ind_split, :] = np.logical_not(split_number == ind_split)
    return training_set_masks

dt = 0.01

experiment_id = 715887471
data = fit_tools.get_data(experiment_id, save_dir='../../example_data')

(licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
 running_speed, running_timestamps, running_acceleration, timebase,
 time_start, time_end) = mo.bin_data(data, dt, time_start=300, time_end=1000)


model = mo.Model(dt=0.01,
              licks=licks_vec, 
              verbose=True,
              name='{}'.format(experiment_id),
              l2=0)

long_lick_filter = mo.MixedGaussianBasisFilter(data = licks_vec,
                                            dt = model.dt,
                                            **filters.long_lick_mixed)
#  model.add_filter('post_lick_mixed', long_lick_filter)

reward_filter = mo.GaussianBasisFilter(data = rewards_vec,
                                    dt = model.dt,
                                    **filters.long_reward)
#  model.add_filter('reward', reward_filter)

flash_filter = mo.GaussianBasisFilter(data = flashes_vec,
                                   dt = model.dt,
                                   **filters.flash)
model.add_filter('flash', flash_filter)

change_filter = mo.GaussianBasisFilter(data = change_flashes_vec,
                                    dt = model.dt,
                                    **filters.change)
#  model.add_filter('change_flash', change_filter)

l2_test = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
#  l2_test = [0, 0.001, 0.01, 0.1, 1, 10]
n_folds = 2
masks = training_set_masks(licks_vec, n_folds)
all_train_nll = np.empty((len(l2_test), n_folds))
all_test_nll = np.empty((len(l2_test), n_folds))

for ind_l2, this_l2 in enumerate(l2_test):
    for ind_fold in range(n_folds):
        print("Fold {}".format(ind_fold))
        train_mask = masks[ind_fold, :]
        model.initialize_filters() # zero out all filter params
        model.fit(bins_to_use=train_mask, l2=this_l2)
        nll_train = model.nll(bins_to_use = train_mask)
        nll_test = model.nll(bins_to_use = np.logical_not(train_mask))
        all_train_nll[ind_l2, ind_fold] = nll_train
        all_test_nll[ind_l2, ind_fold] = nll_test

plt.clf()
test = all_test_nll.sum(axis=1)
train = all_train_nll.sum(axis=1)

plt.plot(list(range(len(l2_test))), test, label='test')
plt.plot(list(range(len(l2_test))), train, label='train')

ax = plt.gca()
ax.set_xticks(list(range(len(l2_test))))
ax.set_xticklabels(["{}".format(l2) for l2 in l2_test])
plt.xlabel('amplitude of L2 penalty')
