import os
import sys
import numpy as onp
import time
import datetime
import h5py
from collections import OrderedDict
from scipy.optimize import minimize
import matplotlib as mpl
#mpl.use('pdf')
from matplotlib import pyplot as plt
import fit_tools

import pickle
import jax.numpy as np
from jax import grad, jit, jacfwd, jacrev, lax

import copy
import itertools

def boxoff(ax, keep="left", yaxis=True):
    """
    Hide axis lines, except left and bottom.
    You can specify which axes to keep: 'left' (default), 'right', 'none'.
    """
    ax.spines["top"].set_visible(False)
    xtlines = ax.get_xticklines()
    ytlines = ax.get_yticklines()
    if keep == "left":
        ax.spines["right"].set_visible(False)
    elif keep == "right":
        ax.spines["left"].set_visible(False)
    elif keep == "none":
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        for t in xtlines + ytlines:
            t.set_visible(False)
    for t in xtlines[1::2] + ytlines[1::2]:
        t.set_visible(False)
    if not yaxis:
        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ytlines = ax.get_yticklines()
        for t in ytlines:
            t.set_visible(False)

def _loglikelihood(licks_vector, latent):
    '''
    Compute the negative log likelihood of poisson observations, given a latent vector

    Args:
        licksdt: a vector of len(time_bins) with 1 if the mouse licked
                 at in that bin
        latent: a vector of the estimated lick rate in each time bin
        params: a vector of the parameters for the model
        l2: amplitude of L2 regularization penalty
    
    Returns: NLL of the model
    '''
    # If there are any zeros in the latent model, have to add "machine tolerance"
    #  latent[latent==0] += np.finfo(float).eps

    # TODO: That wasn't working with autograd, so just add to the whole thing.
    latent += np.finfo(float).eps

    # Get the indices of bins with licks
    licksdt = np.where(licks_vector==1, 1, 0)

    LL = np.sum(np.log(latent)[licksdt]) - np.sum(latent)
    return LL
loglikelihood = jit(_loglikelihood)

def _negative_log_evidence(licks_vector, latent,
                          params,l2=0):
   LL = loglikelihood(licks_vector, latent)
   prior = l2*np.sum(np.array(params)**2)
   log_evidence = LL - prior
   return -1 * log_evidence
negative_log_evidence = jit(_negative_log_evidence)

class Model(object):

    def __init__(self, dt, licks, name=None, l2=0, verbose=False):

        # TODO: Can we use licks as 0/1 vec instead of inds?
        '''
        Args:
            licks: a vector of lick times in dt-index points
        '''

        self.dt = dt
        self.licks = licks
        self.filters = OrderedDict()
        self.latent = None
        self.NLL = None
        self.BIC = None
        self.l2=l2
        self.verbose=verbose

        # Initial param guess for mean rate
        self.mean_rate_param = -0.5
        self.num_time_bins = len(licks)

        if name is None:
            self.name='test'
        else:
            self.name=name

    def add_filter(self, filter_name, filter):
        '''
        Add a filter to the model. 

        Args: 
            filter_name (str): The filter's name
            filter (instance of Filter, GaussianBasisFilter, etc.)
        '''
        self.filters[filter_name] = filter

    def set_filter_params(self, flat_params):
        '''
        Break up a flat array of params and set them for each filter in the model.
        '''
        self.mean_rate_param = flat_params[0] # The first param is always the mean.
        flat_params = flat_params[1:]
        param_start = 0
        for filter_name, filter in self.filters.items():
            num_params = filter.num_params
            filter.set_params(flat_params[param_start:param_start+num_params])
            param_start += num_params
        if not param_start == len(flat_params):
            raise ValueError("We didn't use all of the params when setting")

    def get_filter_params(self):
        '''
        Take params from each filter out into a flat array. 
        '''
        paramlist = [np.array([self.mean_rate_param])] # First param is always the mean rate
        for filter_name, filter in self.filters.items():
            paramlist.append(filter.params)
        return np.concatenate(paramlist)

    def calculate_latent(self):
        '''
        Filters own their params and data, so we just call the linear_output
        method on each filter and add up the result
        '''

        base = np.zeros(self.num_time_bins)
        base += self.mean_rate_param # Add in the mean rate

        for filter_name, filter in self.filters.items():
            base += filter.linear_output()

        latent = np.exp(np.clip(base, -700, 700))
        NLE = negative_log_evidence(self.licks,
                                    latent,
                                    self.get_filter_params(),
                                    self.l2)
        return NLE, latent

    def ll(self):
        '''
        Return the log-liklihood of the data given the model
        '''
        base = np.zeros(self.num_time_bins)
        base += self.mean_rate_param # Add in the mean rate

        for filter_name, filter in self.filters.items():
            base += filter.linear_output()

        latent = np.exp(np.clip(base, -700, 700))
        LL = loglikelihood(self.licks, latent)
        return LL, latent

    # Function to minimize
    def nle(self, params):
        self.set_filter_params(params)
        return self.calculate_latent()[0]
    
    def fit(self):

        params = self.get_filter_params()

        sys.stdout.write("Fitting model with {} params\n".format(len(params)))
        start_time = time.time()

        def print_NLE_callback(xk):
            '''
            A callback for printing info about each iteration.
        
            Args:
                xk: This is the vector of current params (this is how the 
                    callable is executed per the minimize docs). We don't 
                    use it in this func though.
            '''
            sys.stdout.flush() # This and the \r make it keep writing the same line
            sys.stdout.write('\r')
            NLL, latent = self.calculate_latent()
            self.iteration+=1
            sys.stdout.write("Iteration: {} NLE: {}".format(self.iteration, NLL))
            sys.stdout.flush()
        
        kwargs = {}
        if self.verbose:
            self.iteration=0
            kwargs.update({'callback':print_NLE_callback})
        #  res = minimize(wrapper_func, params, **kwargs)

        g = jit(grad(self.nle))
        res = minimize(self.nle, params, jac=g, **kwargs)

        elapsed_time = time.time() - start_time
        sys.stdout.write('\n')
        sys.stdout.write("Done! Elapsed time: {:02f} sec".format(time.time()-start_time))

        # Set the final version of the params for the filters
        self.set_filter_params(res.x)
        self.res = res

    def dropout_analysis(self):
        ll = self.ll()[0]
        dropout_ll_percent_change = {}
        dropout_ll = {}
        if len(self.filters)<1:
            # TODO: Should we define this? Like, drop out the baseline rate?
            print("no filters to drop out")
            pass
        else:
            for ind_filter, filter_to_drop in enumerate(self.filters.keys()):
            # for filters_to_use in itertools.combinations(filter_names,

                #Make a copy of the model
                sub_model = copy.deepcopy(self)

                #Remove the filter
                print("\nRemoving filter: {}\n".format(filter_to_drop))
                del(sub_model.filters[filter_to_drop])

                sub_model.fit()
                sub_ll = sub_model.ll()[0]
                dropout_ll[filter_to_drop] = sub_ll
                dropout_ll_percent_change[filter_to_drop] = 100*((ll-sub_ll)/ll)

        print("Done with dropout analysis")
        self.ll = ll
        self.dropout_ll = dropout_ll
        self.dropout_ll_percent_change = dropout_ll_percent_change

    def eval(self):
        '''
        Evaluate the model, updating the model res with BIC

        '''
        self.res = evaluate_model(self.res, self.calculate_latent,
                                  self.licksdt, self.stop_time)

    def plot_filters(self, show=True, save_dir=None):
        #plt.clf()
        n_filters = len(self.filters)
        for ind_filter, (filter_name, filter_obj) in enumerate(self.filters.items()):
            linear_filt = filter_obj.build_filter()
            plt.subplot(2, (n_filters/2)+1, ind_filter+1)
            plt.plot(filter_obj.filter_time_vec, 
                     np.exp(linear_filt),
                     'k-')
            plt.title(filter_name)
            plt.xlabel('time (s)')
            ax = plt.gca()
            boxoff(ax)
        plt.tight_layout()
        if save_dir is not None:
            fig_name = "{}_nonlinear_filters.pdf".format(self.name)
            plt.savefig(os.path.join(save_dir, fig_name))
        if show:
            plt.show()

    def save(self, output_dir, fn=None, verbose=True):
        if fn==None:
            today = datetime.datetime.now().strftime("%Y%m%d")
            fn = "model_{}_fit_{}.pkl".format(self.name, 
                                              today)
        full_path = os.path.join(output_dir, fn)
        if verbose:
            print("Saving model to: {}".format(full_path))
        pickle_model(self, full_path)

    def save_filter_params(self, output_dir, fn=None):
        if fn==None:
            fn = "model_{}_{}_saved_params.npz".format(self.name, 
                                                       len(self.get_filter_params()))

        param_dict = {filt_name:filt.params for filt_name, filt in self.filters.items()}
        full_path = os.path.join(output_dir, fn)
        np.savez(full_path,
                 mean_rate=self.mean_rate_param,
                 **param_dict)

    def set_filter_params_from_file(self, save_fn):
        # TODO: some smarter checking here. Up to us to not break it.
        param_dict = np.load(save_fn)
        self.mean_rate_param = param_dict['mean_rate']
        for filter_name, filt in self.filters.items():
            filt.set_params(param_dict[filter_name])


# Can we do the convolve func with a vec of 0/1 instead of rolling our own?
# so linear_whatever funcs can use the same thing.

class Filter(object):

    def __init__(self, num_params, data, initial_params=None):
        '''
        Base class for filter objects

        Args:
            num_params (int): Number of filter parameters
            data (np.array): The data relevant to the filter. Always has to
                             be num_bins in length. For filters that operate
                             on discrete events, pass an array with ones at
                             time bin indices where the event happened, and 
                             zero otherwise.
            initial_params (np.array): Initial parameter guesses. 
        '''
        self.num_params = num_params
        self.data = data

        if initial_params is not None:
            self.set_params(initial_params)
        else:
            self.initialize_params()

    def set_params(self, params):
        if not len(params) == self.num_params:
            raise ValueError(("Trying to give {} params to the {} filter"
                              " which takes {} params".format(len(params),
                                                              self.name,
                                                              self.num_params)))
        else:
            self.params = params

    def initialize_params(self):
        '''
        Init all params to zero by default
        '''
        self.params = np.zeros(self.num_params)

    def linear_output(self):
        '''
        This base class just convolves the filter params with the data.
        Cuts the output to be the same length as the data
        '''
        filt = self.build_filter()
        lhs = self.data.reshape(1, 1, -1, 1)
        rhs = filt.reshape(1, 1, -1, 1)
        window_strides = np.array([1, 1])
        padding = 'SAME'
        output = lax.conv_general_dilated(lhs, rhs, window_strides, padding).ravel()[:self.stop_time]

        #  output = np.convolve(self.data, self.params)[:self.stop_time]

        # Shift prediction forward by one time bin
        #  output = np.r_[0, output[:-1]]
        return output


class GaussianBasisFilter(object):
    def __init__(self, num_params, data, dt, duration, sigma):
        '''
        A filter implemented as the sum of a number of gaussians. 

        Param value controls the amplitude of each gaussian. 
        Uniformly spaced over the duration of the filter.
        
        Args:
            num_params (int): Number of gaussians to use
            data (np.array): See documentation for data in Filter
            dt (float): Time in seconds per time bin
            duration (float): filter duration in seconds
            sigma (float): Std for each gaussian.
        '''
        self.name=''
        self.num_params = num_params
        self.data = data
        #  self.set_params(initial_params)
        self.initialize_params()

        #  if initial_params is not None:
        #      self.set_params(initial_params)
        #  else:

        self.duration = duration
        self.dt = dt
        self.sigma = sigma
        self.filter_time_vec = np.arange(dt, duration, dt)


    def set_params(self, params):
        if not len(params) == self.num_params:
            raise ValueError(("Trying to give {} params to the a filter"
                              " which takes {} params".format(len(params),
                                                              self.num_params)))
        else:
            self.params = params

    def initialize_params(self):
        '''
        Init all params to zero by default
        '''
        self.params = np.zeros(self.num_params)



    def build_filter(self):
        '''
        This replicates the function in fit_tools just to make this self-contained.
        '''

        def gaussian_template(x, mu, sigma):
            return (1 / (np.sqrt(2 * 3.14 * sigma ** 2))) * \
                   np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

        mean = (self.filter_time_vec[-1] - self.filter_time_vec[0]) / \
               (self.num_params-1)

        # Empty array to save each gaussian
        basis_funcs = np.empty((self.num_params, len(self.filter_time_vec)))

        # Zero filter array to start with
        filter = np.zeros(np.shape(self.filter_time_vec)) 

        # Add each gaussian to the filter
        for ind_param in range(0, len(self.params)):
            this_gaussian = self.params[ind_param] * \
                            gaussian_template(self.filter_time_vec,
                                              mean * ind_param,
                                              self.sigma)    
            filter += this_gaussian

            # Save this gaussian
            # TODO: Not working with autograd
            # basis_funcs[ind_param, :] = this_gaussian

        return filter

    def linear_output(self):
        filt = self.build_filter()
        output = linear_output_external(self.data, filt)
        return output

def _linear_output_external(data, filt):
    rhs = filt.reshape(1, 1, -1, 1)
    lhs = data.astype(float).reshape(1, 1, -1, 1)
    window_strides = np.array([1, 1])
    padding = 'SAME'
    output = lax.conv_general_dilated(lhs, rhs, window_strides, padding).ravel()[:data.shape[0]]
    # Shift prediction forward by one time bin
    #  output = np.r_[0, output[:-1]]
    output = np.concatenate([np.array([0]), output[:-1]])
    return output
linear_output_external = jit(_linear_output_external)


def bin_data(data, dt, time_start=None, time_end=None):

    lick_timestamps = data['lick_timestamps']
    running_timestamps = data['running_timestamps']
    running_speed = data['running_speed']
    reward_timestamps = data['reward_timestamps']
    flash_timestamps = data['stim_on_timestamps']

    if time_start is None:
        time_start = 0
    if time_end is None:
        time_end = running_timestamps[-1]+0.00001

    change_flash_timestamps = fit_tools.extract_change_flashes(data)

    running_speed = running_speed[(running_timestamps >= time_start) & \
                                  (running_timestamps < time_end)]

    # Normalize running speed
    running_speed = running_speed / running_speed.max()

    running_timestamps = running_timestamps[(running_timestamps >= time_start) & \
                                            (running_timestamps < time_end)]


    reward_timestamps = reward_timestamps[(reward_timestamps >= time_start) & \
                                          (reward_timestamps < time_end)]


    lick_timestamps = lick_timestamps[(lick_timestamps >= time_start) & \
                                      (lick_timestamps < time_end)]

    flash_timestamps = flash_timestamps[(flash_timestamps >= time_start) & \
                                        (flash_timestamps < time_end)]

    change_flash_timestamps = change_flash_timestamps[
        (change_flash_timestamps > time_start) & (change_flash_timestamps < time_end)
    ]

    running_acceleration = fit_tools.compute_running_acceleration(running_speed)


    for arr in [running_timestamps, reward_timestamps, lick_timestamps,
                flash_timestamps, change_flash_timestamps]:
        arr -= running_timestamps[0]

    timebase = onp.arange(0, time_end-time_start, dt)
    edges = onp.arange(0, time_end-time_start+dt, dt)

    licks_vec, _ = onp.histogram(lick_timestamps, bins=edges)
    rewards_vec, _ = onp.histogram(reward_timestamps, bins=edges)
    flashes_vec, _ = onp.histogram(flash_timestamps, bins=edges)
    change_flashes_vec, _ = onp.histogram(change_flash_timestamps, bins=edges)

    return (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
            running_speed, running_timestamps, running_acceleration,
            timebase, time_start, time_end)

# Funcs for saving and loading models with pickle
def pickle_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def unpickle_model(filename):
    with open(filename, 'rb') as f:
        model = pickle.load(f)
        return model

if __name__ == "__main__":

    dt = 0.01

    experiment_id = 715887471
    data = fit_tools.get_data(experiment_id, save_dir='../example_data')

    (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
     running_speed, running_timestamps, running_acceleration, timebase,
     time_start, time_end) = bin_data(data, dt, time_start=300, time_end=1000)

    #  (licks_vec, rewards_vec, flashes_vec, change_flashes_vec,
    #   running_speed, running_timestamps, running_acceleration, timebase,
    #   time_start, time_end) = bin_data(data, dt)

    case=7
    if case==0:
        # Model with just mean rate param
        model = Model(dt=0.01,
                      licks=licks_vec,
                      verbose=True)
        model.fit()

    elif case==1:

        model = Model(dt=0.01,
                      licks=licks_vec,
                      verbose=True)

        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec,
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        model.add_filter('post_lick', post_lick_filter)

        #print(grad(model.set_params_and_calculate_nll)(model.get_filter_params()))

        model.fit()


    elif case==2:

        model = Model(dt=0.01,
                      licks=licks_vec, 
                      verbose=True,
                      l2=0.1)
        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec.astype(np.float),
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        model.add_filter('post_lick', post_lick_filter)

        reward_filter = GaussianBasisFilter(num_params = 10,
                                            data = rewards_vec.astype(np.float),
                                            dt = model.dt,
                                            duration = 4,
                                            sigma = 0.50)
        model.add_filter('reward', reward_filter)

        model.fit()

    elif case==3:

        model = Model(dt=0.01,
                      licks=licks_vec,
                      verbose=True)
        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec,
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        #  model.add_filter('post_lick', post_lick_filter)

        flash_filter = GaussianBasisFilter(num_params = 15,
                                            data = flashes_vec,
                                            dt = model.dt,
                                            duration = 0.76,
                                            sigma = 0.05)
        model.add_filter('flash', flash_filter)

        reward_filter = GaussianBasisFilter(num_params = 20,
                                            data = rewards_vec,
                                            dt = model.dt,
                                            duration = 4,
                                            sigma = 0.25)
        model.add_filter('reward', reward_filter)

        model.fit()

    elif case==4:

        model = Model(dt=0.01,
                      licks=licks_vec)

        change_filter = GaussianBasisFilter(num_params = 30,
                                            data = change_flashes_vec,
                                            dt = model.dt,
                                            duration = 1.6,
                                            sigma = 0.05)
        model.add_filter('change_flash', change_filter)

        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec,
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        model.add_filter('post_lick', post_lick_filter)


        model.fit()

        # running_speed_filter = Filter(num_params = 6,
        #                               data = running_speed)
        # model.add_filter('running_speed', running_speed_filter)

    elif case==5:
        import filters
        import importlib; importlib.reload(filters)

        model = Model(dt=0.01,
                      licks=licks_vec, 
                      verbose=True,
                      l2=0.5)

        post_lick_filter = GaussianBasisFilter(data = licks_vec,
                                               dt = model.dt,
                                               **filters.post_lick)
        model.add_filter('post_lick', post_lick_filter)

        reward_filter = GaussianBasisFilter(data = rewards_vec,
                                            dt = model.dt,
                                            **filters.reward)
        model.add_filter('reward', reward_filter)

        flash_filter = GaussianBasisFilter(data = flashes_vec,
                                           dt = model.dt,
                                           **filters.flash)
        model.add_filter('flash', flash_filter)

        change_filter = GaussianBasisFilter(data = change_flashes_vec,
                                            dt = model.dt,
                                            **filters.change)
        model.add_filter('change_flash', change_filter)

        running_speed_filter= GaussianBasisFilter(data = running_speed,
                                                  dt = model.dt,
                                                  **filters.running_speed)
        model.add_filter('running_speed', running_speed_filter)

        acceleration_filter= GaussianBasisFilter(data = running_acceleration,
                                                 dt = model.dt,
                                                 **filters.acceleration)
        model.add_filter('acceleration', acceleration_filter)

        model.fit()

    elif case==6:

        import filters
        import importlib; importlib.reload(filters)

        param_save_fn = '/allen/programs/braintv/workgroups/nc-ophys/nick.ponvert/data/model_test_71_saved_params.npz'

        model = Model(dt=0.01,
                      licks=licks_vec, 
                      verbose=True,
                      l2=0.5)

        post_lick_filter = GaussianBasisFilter(data = licks_vec,
                                               dt = model.dt,
                                               **filters.post_lick)
        model.add_filter('post_lick', post_lick_filter)

        reward_filter = GaussianBasisFilter(data = rewards_vec,
                                            dt = model.dt,
                                            **filters.reward)
        model.add_filter('reward', reward_filter)

        flash_filter = GaussianBasisFilter(data = flashes_vec,
                                           dt = model.dt,
                                           **filters.flash)
        model.add_filter('flash', flash_filter)

        change_filter = GaussianBasisFilter(data = change_flashes_vec,
                                            dt = model.dt,
                                            **filters.change)
        model.add_filter('change_flash', change_filter)

        running_speed_filter= GaussianBasisFilter(data = running_speed,
                                                  dt = model.dt,
                                                  **filters.running_speed)
        model.add_filter('running_speed', running_speed_filter)

        acceleration_filter= GaussianBasisFilter(data = running_acceleration,
                                                 dt = model.dt,
                                                 **filters.acceleration)
        model.add_filter('acceleration', acceleration_filter)
        # model.set_filter_params_from_file(param_save_fn)
        model.fit()

    elif case==7:

        import filters
        import importlib; importlib.reload(filters)

        model = Model(dt=0.01,
                      licks=licks_vec, 
                      verbose=True,
                      name='{}'.format(experiment_id),
                      l2=0.5)

        #  post_lick_filter = mo.GaussianBasisFilter(data = licks_vec,
        #                                         dt = model.dt,
        #                                         **filters.post_lick)
        #  model.add_filter('post_lick', post_lick_filter)

        long_lick_filter = GaussianBasisFilter(data = licks_vec,
                                               dt = model.dt,
                                               **filters.long_lick)
        model.add_filter('post_lick', long_lick_filter)

        reward_filter = GaussianBasisFilter(data = rewards_vec,
                                            dt = model.dt,
                                            **filters.reward)
        model.add_filter('reward', reward_filter)

        #  long_reward_filter = mo.GaussianBasisFilter(data = rewards_vec,
        #                                      dt = model.dt,
        #                                      **filters.long_reward)
        #  model.add_filter('long_reward', long_reward_filter)

        flash_filter = GaussianBasisFilter(data = flashes_vec,
                                           dt = model.dt,
                                           **filters.flash)
        model.add_filter('flash', flash_filter)

        change_filter = GaussianBasisFilter(data = change_flashes_vec,
                                            dt = model.dt,
                                            **filters.change)
        model.add_filter('change_flash', change_filter)

        model.fit()

    elif case==8:

        model = Model(dt=0.01,
                      verbose=True,
                      licks=licks_vec)

        change_filter = GaussianBasisFilter(num_params = 30,
                                            data = change_flashes_vec,
                                            dt = model.dt,
                                            duration = 1.6,
                                            sigma = 0.05)
        model.add_filter('change_flash', change_filter)

        post_lick_filter = GaussianBasisFilter(num_params = 10,
                                               data = licks_vec,
                                               dt = model.dt,
                                               duration = 0.21,
                                               sigma = 0.025)
        model.add_filter('post_lick', post_lick_filter)

        model.fit()
        ll = model.ll()[0]
        dropout_ll = model.dropout_analysis()

'''
def save_model_params(model):
    # db_group = h5py.require_group("/")
    f = h5py.File('model_test.h5', 'a')
    model_grp = f.create_group("model")

    model_attrs = list(model.__dict__.keys())
    model_attrs.remove('filters') # We will save filters separately
    for attr in model_attrs:
        data = getattr(model, attr)
        if not isinstance(data, np.ndarray):
            data = np.array([data])
        dset_attr = model_grp.create_dataset(attr, data, dtype=data.dtype)

    for filter_name, filter in model.filters.items():
        filter_grp = f.create_group(filter_name)

        filter_attrs = list(filter.__dict__.keys())
        for attr in filter_attrs:
            data = getattr(filter, attr)
            dset_attr = filter_grp.create_dataset(attr, data, dtype=data.dtype)

'''
